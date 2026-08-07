import gc
import os
import sys
import re
import io
import json
import zipfile
import tempfile
from collections import OrderedDict
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from typing import Dict, Any, Optional, Tuple

# Each cached (symbol, model_type) pair holds a full Keras model in memory
# with no natural upper bound - on a memory-constrained host (e.g. Render's
# free tier), caching every distinct symbol/model a demo touches eventually
# OOMs the process. Capping the cache size and evicting least-recently-used
# entries keeps memory bounded without changing any prediction behavior.
MAX_CACHED_MODELS = int(os.getenv("MAX_CACHED_MODELS", "2"))

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from .utils import add_technical_indicators, fetch_last_60_minutes
    from .nse_data_fetcher import NSEDataFetcher
    from .news_sentiment import get_daily_sentiment_series
except ImportError:
    # Fallback for standalone execution
    from utils import add_technical_indicators, fetch_last_60_minutes
    from nse_data_fetcher import NSEDataFetcher
    from news_sentiment import get_daily_sentiment_series

# Directory and file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

FEATURES = [
    'open', 'high', 'low', 'volume',
    'sma_10', 'ema_9', 'ema_21', 'rsi_14', 'ma_20',
    'bb_upper', 'bb_lower', 'ema_12', 'ema_26',
    'macd', 'signal_line', '%k', '%d', 'atr_14',
    'news_sentiment'
]

# Asset type definitions
STOCK_SYMBOLS = [
    "RELIANCE.NS", "AXISBANK.NS", "HDFCBANK.NS", "ONGC.NS", "SBIN.NS",
    "INFY.NS", "TCS.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "ADANIPORTS.NS",
    "ADANIENT.NS", "BAJFINANCE.NS", "BHARTIARTL.NS"
]

INDEX_SYMBOLS = [
    "^NSEI", "^NSEBANK", "^NSEMDCP50", "^CNXAUTO"
]


# ---------------------------------------------------------------------------
# Keras version-compatible model loader
# Models were saved in a different Keras minor version that included
# 'renorm' (BatchNormalization) and 'use_gate' (MultiHeadAttention) kwargs
# that no longer exist. We patch the config.json inside the .keras zip before
# loading so Keras doesn't reject the files.
# ---------------------------------------------------------------------------
def _strip_compat_keys(cfg):
    """Recursively strip version-specific layer kwargs from a config dict."""
    if isinstance(cfg, dict):
        inner = cfg.get("config", {})
        cls = cfg.get("class_name", "")
        if cls == "BatchNormalization":
            for k in ("renorm", "renorm_clipping", "renorm_momentum"):
                inner.pop(k, None)
        if cls == "MultiHeadAttention":
            inner.pop("use_gate", None)
        for v in cfg.values():
            _strip_compat_keys(v)
    elif isinstance(cfg, list):
        for item in cfg:
            _strip_compat_keys(item)


def _load_keras_model(path: str):
    """
    Load a .keras model robustly across Keras minor version differences.
    Fast path: direct load. Fallback: patch config.json inside the zip.
    """
    # Fast path — works when Keras versions match exactly
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        pass

    # Fallback: open the .keras zip, strip incompatible keys, reload
    try:
        buf = io.BytesIO()
        with zipfile.ZipFile(path, "r") as zin, \
             zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                fname = item.filename
                data = zin.read(fname)
                if fname == "config.json":
                    cfg = json.loads(data.decode("utf-8"))
                    _strip_compat_keys(cfg)
                    data = json.dumps(cfg).encode("utf-8")
                zout.writestr(item, data)

        buf.seek(0)
        tmp = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
        tmp.write(buf.read())
        tmp.close()
        try:
            model = tf.keras.models.load_model(tmp.name, compile=False)
        finally:
            os.unlink(tmp.name)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path}: {e}") from e

class AssetAwarePredictor:
    def __init__(self):
        self.nse_fetcher = NSEDataFetcher()
        self.model_cache = OrderedDict()
        self.scaler_cache = OrderedDict()
        self.metadata_cache = OrderedDict()
        
        # Asset type configurations for validation
        self.asset_configs = {
            'stocks': {
                'price_range': (50, 15000),       # wide enough for HDFC/Airtel/Adani
                'max_prediction_deviation': 0.3,
                'sequence_length': 120
            },
            'indices': {
                'price_range': (5000, 100000),    # covers Nifty to BankNifty
                'max_prediction_deviation': 0.15,
                'sequence_length': 120
            }
        }

    def detect_asset_type(self, symbol):
        """Detect if symbol is a stock or index"""
        if symbol.startswith("^"):
            return "indices"
        elif symbol in INDEX_SYMBOLS:
            return "indices"
        else:
            return "stocks"

    def get_model_paths(self, symbol, model_type):
        """Get file paths for model and scalers"""
        asset_type = self.detect_asset_type(symbol)
        symbol_clean = symbol.replace('.', '_').replace('^', 'INDEX_')
        
        model_filename = f"{model_type}_{asset_type}_{symbol_clean}_model.keras"
        feature_scaler_filename = f"feature_scaler_{asset_type}_{symbol_clean}.pkl"
        target_scaler_filename = f"target_scaler_{asset_type}_{symbol_clean}.pkl"
        metadata_filename = f"metadata_{model_type}_{asset_type}_{symbol_clean}.json"
        legacy_metadata_filename = f"metadata_{asset_type}_{symbol_clean}.json"
        
        return {
            'model': os.path.join(MODEL_DIR, model_filename),
            'feature_scaler': os.path.join(MODEL_DIR, feature_scaler_filename),
            'target_scaler': os.path.join(MODEL_DIR, target_scaler_filename),
            'metadata': os.path.join(MODEL_DIR, metadata_filename),
            'legacy_metadata': os.path.join(MODEL_DIR, legacy_metadata_filename)
        }

    def get_sequence_length(self, symbol, model_type, default_seq_len):
        cache_key = f"{symbol}_{model_type}"
        metadata = self.metadata_cache.get(cache_key, {})
        sequence_length = metadata.get('sequence_length', default_seq_len)
        try:
            return int(sequence_length)
        except (TypeError, ValueError):
            return int(default_seq_len)

    def load_asset_model(self, symbol, model_type):
        """Load asset-specific model and scalers"""
        cache_key = f"{symbol}_{model_type}"
        
        if cache_key in self.model_cache:
            self.model_cache.move_to_end(cache_key)
            return self.model_cache[cache_key], self.scaler_cache[cache_key]
        
        paths = self.get_model_paths(symbol, model_type)
        
        try:
            required_paths = [paths['model'], paths['feature_scaler'], paths['target_scaler']]

            # Check if required model artifacts exist
            if all(os.path.exists(path) for path in required_paths):
                print(f"[INFO] Loading asset-specific {model_type} model for {symbol}")

                model = _load_keras_model(paths['model'])
                feature_scaler = joblib.load(paths['feature_scaler'])
                target_scaler = joblib.load(paths['target_scaler'])

                metadata = {}
                metadata_path = paths['metadata'] if os.path.exists(paths['metadata']) else paths['legacy_metadata']
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                else:
                    print(f"[WARNING] Metadata file missing for {symbol} {model_type}, using defaults")
                
                # Cache the loaded components
                self.model_cache[cache_key] = model
                self.scaler_cache[cache_key] = (feature_scaler, target_scaler)
                self.metadata_cache[cache_key] = metadata
                self._evict_lru_if_needed()

                print(f"[INFO] ✅ Asset-specific model loaded for {symbol}")
                return model, (feature_scaler, target_scaler)
            
            else:
                print(f"[WARNING] Asset-specific model not found for {symbol}")
                return self.load_fallback_model(model_type)
                
        except Exception as e:
            print(f"[ERROR] Failed to load asset-specific model for {symbol}: {str(e)}")
            return self.load_fallback_model(model_type)

    def _evict_lru_if_needed(self):
        """Keep at most MAX_CACHED_MODELS loaded, dropping the least-recently-used."""
        while len(self.model_cache) > MAX_CACHED_MODELS:
            evicted_key, _ = self.model_cache.popitem(last=False)
            self.scaler_cache.pop(evicted_key, None)
            self.metadata_cache.pop(evicted_key, None)
            print(f"[INFO] Evicted cached model for {evicted_key} to bound memory usage")
        gc.collect()

    def load_fallback_model(self, model_type):
        """Load fallback improved models"""
        try:
            if model_type == "lstm":
                model_path = os.path.join(MODEL_DIR, "lstm_stock_model_improved.keras")
            else:
                model_path = os.path.join(MODEL_DIR, "transformer_stock_model_improved.keras")
            
            feature_scaler_path = os.path.join(MODEL_DIR, "feature_scaler_improved.pkl")
            target_scaler_path = os.path.join(MODEL_DIR, "target_scaler_improved.pkl")
            
            if all(os.path.exists(path) for path in [model_path, feature_scaler_path, target_scaler_path]):
                model = _load_keras_model(model_path)
                feature_scaler = joblib.load(feature_scaler_path)
                target_scaler = joblib.load(target_scaler_path)
                
                print(f"[INFO] ✅ Fallback {model_type} model loaded")
                return model, (feature_scaler, target_scaler)
            else:
                print(f"[ERROR] Fallback {model_type} model not found")
                return None, (None, None)
                
        except Exception as e:
            print(f"[ERROR] Failed to load fallback {model_type} model: {str(e)}")
            return None, (None, None)

    def preprocess_data_consistent(self, df, asset_type, symbol, news_sentiment_value=None):
        """Consistent preprocessing matching training pipeline"""
        # Convert MultiIndex or ticker-suffixed columns to simple names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns.values]
        
        # Remove ticker suffix
        df.columns = [re.sub(r'_[A-Z^]+[A-Z0-9]*\.?[A-Z]*$', '', col) for col in df.columns]
        
        # Ensure lowercase columns
        df.columns = df.columns.str.lower()
        
        # Add technical indicators
        df = add_technical_indicators(df)
        if df is None or df.empty:
            print("[ERROR] Failed to add technical indicators")
            return None
            
        df = df.ffill().bfill()
        
        # Build date-aligned sentiment series (consistent with training).
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date']).dt.normalize()
        elif 'datetime' in df.columns:
            dates = pd.to_datetime(df['datetime']).dt.normalize()
        elif isinstance(df.index, pd.DatetimeIndex):
            dates = pd.to_datetime(df.index).normalize()
        else:
            dates = None

        if dates is not None:
            start_date = dates.min().date()
            end_date = dates.max().date()
            daily_sentiment = get_daily_sentiment_series(
                symbol_or_query=symbol,
                start_date=start_date,
                end_date=end_date,
                use_cache=True,
            )
            df['news_sentiment'] = dates.dt.strftime('%Y-%m-%d').map(daily_sentiment).fillna(0.0).astype(float)
        else:
            df['news_sentiment'] = 0.0

        # Optionally override the latest candle sentiment with fresh runtime sentiment.
        if news_sentiment_value is not None and len(df) > 0:
            df.loc[df.index[-1], 'news_sentiment'] = float(news_sentiment_value)

        # Asset-specific volume normalization (matching training)
        if asset_type == "stocks":
            df['volume_pct_change'] = df['volume'].pct_change().fillna(0)
            df['volume_pct_change'] = df['volume_pct_change'].clip(-0.5, 0.5)
        else:
            df['volume_pct_change'] = df['volume'].pct_change().fillna(0)
            df['volume_pct_change'] = df['volume_pct_change'].clip(-0.2, 0.2)
        
        df['volume'] = df['volume_pct_change']

        # Asset-specific price change normalization (matching training)
        price_change_limit = 0.1 if asset_type == "indices" else 0.2
        for col in ['open', 'high', 'low', 'close']:
            df[f'pct_change_{col}'] = df[col].pct_change().fillna(0)
            df[f'pct_change_{col}'] = df[f'pct_change_{col}'].clip(-price_change_limit, price_change_limit)

        df.fillna(0, inplace=True)
        return df

    def validate_prediction(self, prediction, current_price, symbol):
        """Validate prediction against reasonable bounds"""
        asset_type = self.detect_asset_type(symbol)
        config = self.asset_configs[asset_type]
        
        # Check if prediction is within reasonable deviation from current price
        max_deviation = config['max_prediction_deviation']
        min_prediction = current_price * (1 - max_deviation)
        max_prediction = current_price * (1 + max_deviation)
        
        if prediction < min_prediction or prediction > max_prediction:
            print(f"[WARNING] Prediction {prediction:.2f} outside reasonable range [{min_prediction:.2f}, {max_prediction:.2f}]")
            # Clamp prediction to reasonable range
            prediction = np.clip(prediction, min_prediction, max_prediction)
            print(f"[INFO] Clamped prediction to {prediction:.2f}")
        
        # Check if prediction is within asset type price range
        price_min, price_max = config['price_range']
        if prediction < price_min * 0.5 or prediction > price_max * 2:
            print(f"[WARNING] Prediction {prediction:.2f} outside asset type range")
            # Use current price as fallback
            prediction = current_price
            print(f"[INFO] Using current price as fallback: {prediction:.2f}")
        
        return prediction

    def create_sequences(self, X, seq_length=60):
        """Create sequences for prediction"""
        if len(X) < seq_length:
            print(f"[ERROR] Not enough data points: {len(X)} < {seq_length}")
            return np.array([])
        
        X_seq = []
        for i in range(len(X) - seq_length + 1):
            X_seq.append(X[i:i + seq_length])
        return np.array(X_seq)

    def predict_price(self, symbol, model_type="transformer", use_daily_data=True, news_sentiment_value=None):
        """
        Predict stock price using asset-aware models
        
        Args:
            symbol: Stock symbol
            model_type: "lstm" or "transformer"
            use_daily_data: If True, use daily data for consistency with training
        """
        try:
            print(f"[INFO] Predicting {symbol} using {model_type} model")
            
            # Detect asset type
            asset_type = self.detect_asset_type(symbol)
            config = self.asset_configs[asset_type]
            
            # Load asset-specific model and scalers
            model, (feature_scaler, target_scaler) = self.load_asset_model(symbol, model_type)
            
            if model is None or feature_scaler is None or target_scaler is None:
                return {
                    "error": f"Failed to load {model_type} model for {symbol}",
                    "symbol": symbol,
                    "model_type": model_type
                }

            # Fetch data - use daily data for consistency with training
            if use_daily_data:
                print(f"[INFO] Fetching daily data for consistency with training")
                df = self.nse_fetcher.fetch_data(symbol)
            else:
                print(f"[INFO] Fetching intraday data")
                df = fetch_last_60_minutes(symbol)
            
            if df is None or df.empty:
                return {
                    "error": f"No data found for symbol {symbol}",
                    "symbol": symbol,
                    "model_type": model_type
                }

            # Preprocess data consistently with training
            df = self.preprocess_data_consistent(df, asset_type, symbol, news_sentiment_value=news_sentiment_value)
            if df is None or df.empty:
                return {
                    "error": "Failed to preprocess data",
                    "symbol": symbol,
                    "model_type": model_type
                }

            # Prepare features (same as training)
            extended_features = FEATURES + [f'pct_change_{col}' for col in ['open', 'high', 'low', 'close']]
            X = df[extended_features].values
            y = df['close'].values.reshape(-1, 1)

            print(f"[INFO] Data shape: X={X.shape}, y={y.shape}")
            print(f"[INFO] Current price: {y[-1][0]:.2f}")

            # Scale features and target
            X_scaled = feature_scaler.transform(X)
            y_scaled = target_scaler.transform(y)

            # Create sequences
            seq_length = self.get_sequence_length(symbol, model_type, config['sequence_length'])
            X_seq = self.create_sequences(X_scaled, seq_length)
            
            if X_seq.size == 0:
                return {
                    "error": f"Not enough data to create sequences (need {seq_length} points)",
                    "symbol": symbol,
                    "model_type": model_type
                }

            print(f"[INFO] Created {len(X_seq)} sequences for prediction")

            # Make prediction
            predictions_scaled = model.predict(X_seq, verbose=0)
            predictions = target_scaler.inverse_transform(predictions_scaled)

            # Get the latest prediction
            latest_prediction = float(predictions[-1][0])
            current_price = float(y[-1][0])

            # Validate prediction
            validated_prediction = self.validate_prediction(latest_prediction, current_price, symbol)

            # Log raw and validated predictions for debugging
            print(f"[DEBUG] Raw prediction: {latest_prediction:.2f}, Validated prediction: {validated_prediction:.2f}")

            # Calculate confidence based on recent predictions consistency
            if len(predictions) > 1:
                recent_preds = predictions[-5:].flatten()  # Last 5 predictions
                pred_std = np.std(recent_preds)
                pred_mean = np.mean(recent_preds)
                confidence = max(0, min(100, 100 - (pred_std / pred_mean * 100)))
            else:
                confidence = 50.0  # Default confidence

            # Get model metadata if available
            cache_key = f"{symbol}_{model_type}"
            metadata = self.metadata_cache.get(cache_key, {})

            result = {
                "predicted_close": round(float(validated_prediction), 2),
                "actual_close": round(float(current_price), 2),
                "raw_prediction": round(float(latest_prediction), 2),
                "symbol": symbol,
                "model_type": model_type,
                "asset_type": asset_type,
                "confidence": round(float(confidence), 1),
                "data_points_used": int(len(df)),
                "sequences_created": int(len(X_seq)),
                "model_metadata": {
                    "training_mae": float(metadata.get('test_mae', 0)) if metadata.get('test_mae') != 'N/A' else 'N/A',
                    "training_rmse": float(metadata.get('test_rmse', 0)) if metadata.get('test_rmse') != 'N/A' else 'N/A',
                    "price_range": metadata.get('price_range', 'N/A')
                }
            }

            print(f"[INFO] ✅ Prediction completed for {symbol}")
            print(f"[INFO] Current: {current_price:.2f}, Predicted: {validated_prediction:.2f}")
            
            return result

        except Exception as e:
            print(f"[ERROR] Prediction failed for {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"Prediction failed: {str(e)}",
                "symbol": symbol,
                "model_type": model_type
            }

    def compare_models(self, symbol, news_sentiment_value=None):
        """Compare LSTM and Transformer predictions for a symbol with time series data"""
        print(f"\n[INFO] Comparing models for {symbol}")
        
        # Get detailed predictions with time series data
        lstm_result = self.predict_with_time_series(symbol, "lstm", news_sentiment_value=news_sentiment_value)
        transformer_result = self.predict_with_time_series(symbol, "transformer", news_sentiment_value=news_sentiment_value)
        
        return {
            "symbol": symbol,
            "LSTM": lstm_result,
            "Transformer": transformer_result,
            "comparison": {
                "current_price": lstm_result.get("actual_close", "N/A"),
                "lstm_prediction": lstm_result.get("predicted_close", "N/A"),
                "transformer_prediction": transformer_result.get("predicted_close", "N/A"),
                "prediction_difference": abs(
                    lstm_result.get("predicted_close", 0) - 
                    transformer_result.get("predicted_close", 0)
                ) if "predicted_close" in lstm_result and "predicted_close" in transformer_result else "N/A"
            }
        }

    def predict_with_time_series(self, symbol, model_type="transformer", news_sentiment_value=None):
        """
        Predict with time series data for charting and metrics calculation
        """
        try:
            print(f"[INFO] Generating time series predictions for {symbol} using {model_type}")
            
            # Detect asset type
            asset_type = self.detect_asset_type(symbol)
            config = self.asset_configs[asset_type]
            
            # Load asset-specific model and scalers
            model, (feature_scaler, target_scaler) = self.load_asset_model(symbol, model_type)
            
            if model is None or feature_scaler is None or target_scaler is None:
                return {
                    "error": f"Failed to load {model_type} model for {symbol}",
                    "symbol": symbol,
                    "model_type": model_type
                }

            # Fetch daily data for consistency with training
            print(f"[INFO] Fetching daily data for time series analysis")
            df = self.nse_fetcher.fetch_data(symbol)
            
            if df is None or df.empty:
                return {
                    "error": f"No data found for symbol {symbol}",
                    "symbol": symbol,
                    "model_type": model_type
                }

            # Preprocess data consistently with training
            df = self.preprocess_data_consistent(df, asset_type, symbol, news_sentiment_value=news_sentiment_value)
            if df is None or df.empty:
                return {
                    "error": "Failed to preprocess data",
                    "symbol": symbol,
                    "model_type": model_type
                }

            # Prepare features (same as training)
            extended_features = FEATURES + [f'pct_change_{col}' for col in ['open', 'high', 'low', 'close']]
            X = df[extended_features].values
            y = df['close'].values.reshape(-1, 1)

            print(f"[INFO] Data shape: X={X.shape}, y={y.shape}")

            # Scale features and target
            X_scaled = feature_scaler.transform(X)
            y_scaled = target_scaler.transform(y)

            # Create sequences for time series prediction
            seq_length = self.get_sequence_length(symbol, model_type, config['sequence_length'])
            X_seq, y_seq = [], []
            
            for i in range(len(X_scaled) - seq_length):
                X_seq.append(X_scaled[i:i + seq_length])
                y_seq.append(y_scaled[i + seq_length])
            
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            if X_seq.size == 0 or y_seq.size == 0:
                return {
                    "error": f"Not enough data to create sequences (need {seq_length} points)",
                    "symbol": symbol,
                    "model_type": model_type
                }

            print(f"[INFO] Created {len(X_seq)} sequences for time series prediction")

            # Make predictions for all sequences
            predictions_scaled = model.predict(X_seq, verbose=0)
            predictions = target_scaler.inverse_transform(predictions_scaled)
            actuals = target_scaler.inverse_transform(y_seq)

            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            mse = mean_squared_error(actuals, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)

            # Get the latest prediction
            latest_prediction = float(predictions[-1][0])
            current_price = float(actuals[-1][0])

            # Validate prediction
            validated_prediction = self.validate_prediction(latest_prediction, current_price, symbol)

            # Calculate confidence based on recent predictions consistency
            if len(predictions) > 5:
                recent_preds = predictions[-5:].flatten()
                pred_std = np.std(recent_preds)
                pred_mean = np.mean(recent_preds)
                confidence = max(0, min(100, 100 - (pred_std / pred_mean * 100)))
            else:
                confidence = 50.0

            # Get model metadata if available
            cache_key = f"{symbol}_{model_type}"
            metadata = self.metadata_cache.get(cache_key, {})

            # Create time series for charting (use last 30 points for better visualization)
            chart_length = min(30, len(predictions))
            time_series = list(range(chart_length))
            chart_predictions = predictions[-chart_length:].flatten().tolist()
            chart_actuals = actuals[-chart_length:].flatten().tolist()

            result = {
                "predicted_close": round(float(validated_prediction), 2),
                "actual_close": round(float(current_price), 2),
                "raw_prediction": round(float(latest_prediction), 2),
                "symbol": symbol,
                "model_type": model_type,
                "asset_type": asset_type,
                "confidence": round(float(confidence), 1),
                "data_points_used": int(len(df)),
                "sequences_created": int(len(X_seq)),
                
                # Time series data for charting
                "time_series": time_series,
                "predictions": chart_predictions,
                "actuals": chart_actuals,
                
                # Metrics for comparison table
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "r2_score": float(r2),
                
                # Model metadata
                "model_metadata": {
                    "training_mae": float(metadata.get('test_mae', 0)) if metadata.get('test_mae') != 'N/A' else 'N/A',
                    "training_rmse": float(metadata.get('test_rmse', 0)) if metadata.get('test_rmse') != 'N/A' else 'N/A',
                    "price_range": metadata.get('price_range', 'N/A')
                },
                
                # Graph URLs (placeholder for now)
                "performance_graph_url": f"/static/graphs/{model_type}_performance.png",
                "accuracy_graph_url": f"/static/graphs/{model_type}_accuracy.png"
            }

            print(f"[INFO] ✅ Time series prediction completed for {symbol}")
            print(f"[INFO] Current: {current_price:.2f}, Predicted: {validated_prediction:.2f}")
            print(f"[INFO] Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
            
            return result

        except Exception as e:
            print(f"[ERROR] Time series prediction failed for {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"Time series prediction failed: {str(e)}",
                "symbol": symbol,
                "model_type": model_type
            }

    def get_available_models(self):
        """Get list of available asset-specific models"""
        available_models = {}
        
        for symbol in STOCK_SYMBOLS + INDEX_SYMBOLS:
            symbol_models = []
            
            for model_type in ["lstm", "transformer"]:
                paths = self.get_model_paths(symbol, model_type)
                required_paths = [paths['model'], paths['feature_scaler'], paths['target_scaler']]
                if all(os.path.exists(path) for path in required_paths):
                    symbol_models.append(model_type)
            
            if symbol_models:
                available_models[symbol] = symbol_models
        
        return available_models

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol to predict")
    parser.add_argument("--model", type=str, choices=["lstm", "transformer", "both"], default="both", help="Model type")
    parser.add_argument("--daily", action="store_true", help="Use daily data instead of intraday")
    
    args = parser.parse_args()
    
    predictor = AssetAwarePredictor()
    
    if args.model == "both":
        result = predictor.compare_models(args.symbol)
        print(json.dumps(result, indent=2))
    else:
        result = predictor.predict_price(args.symbol, args.model, args.daily)
        print(json.dumps(result, indent=2))
