# Phase 1 Complete: Production Deployment

Status snapshot of the productionization work described in [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) and [`docs/DEPLOYMENT.md`](DEPLOYMENT.md). This is Phase 1 of a three-part track (this project → RAG agent on AWS → new GCP project) converting working ML prototypes into deployed, portfolio-grade systems.

## What shipped

- **Containerized**: multi-stage Docker images for backend (Django + gunicorn, models baked in) and frontend (Vite build → nginx), orchestrated locally via `docker-compose.yml`.
- **Env-driven config & security hardening**: no hardcoded secrets, production-only Django hardening (HSTS, secure cookies, SSL redirect), DRF request throttling, least-privilege CORS.
- **CI/CD**: GitHub Actions (`.github/workflows/ci.yml`) runs backend tests → frontend tests → Docker build + Trivy scan → gated deploys, triggered on push to `main` or manually via `workflow_dispatch`. CodeQL and gitleaks scan every push/PR independently.
- **Deployed**: backend on Render (Docker web service + managed Postgres, via `render.yaml` Blueprint), frontend on Vercel (static Vite build). Both deploys are triggered from GitHub Actions, not the platforms' own auto-deploy, so every deploy is gated on tests/build/scan passing.
- **Live**:
  - Frontend: https://smart-stock-prediction.vercel.app
  - Backend: https://time-series-system-backend.onrender.com

## Real problems hit and fixed along the way

Deploying this for real surfaced several issues that never show up running locally — worth keeping as a record of what "production-ready" actually required, beyond the initial plan:

1. **`os.getenv(KEY, default)` doesn't handle present-but-empty env vars.** A blank `SECRET_KEY=` line in `.env` still set `os.environ['SECRET_KEY'] = ""`, so the intended dev-default fallback never triggered. Fixed by switching to `os.getenv(KEY) or default` for `SECRET_KEY`/`DEBUG`/`ALLOWED_HOSTS`/CORS settings.
2. **`SECURE_SSL_REDIRECT=True` broke Django's test client.** Enabling it whenever `DEBUG=False` (correct for production) also broke CI, which runs tests over plain HTTP with `DEBUG=False` to simulate production config. Made it independently overridable so CI can disable just the redirect.
3. **Render injects `PORT` (default `10000`); gunicorn was hardcoded to `8000`.** Render's health check hit the wrong port and timed out the deploy. Fixed `entrypoint.sh` and the Docker `HEALTHCHECK` to respect `$PORT`.
4. **The container's own health probe hits `127.0.0.1`, which is never in a production `ALLOWED_HOSTS`.** Would have permanently failed Docker's internal `HEALTHCHECK` in production. `localhost`/`127.0.0.1` are now unconditionally allowed alongside whatever `ALLOWED_HOSTS` specifies for real traffic.
5. **Render's Blueprint UI silently drops blank `sync: false` env vars** instead of creating them empty — `ALLOWED_HOSTS` and `CORS_ALLOWED_ORIGINS` had to be added manually after first deploy once the real domains were known.
6. **`aquasecurity/trivy-action@0.28.0` doesn't exist** — the repo's tags use a `v` prefix (`v0.28.0`). A one-character pin fixed a job that failed before running any actual step.
7. **Real memory pressure on Render's free tier (512MB).** Two gunicorn workers each loaded a full copy of TensorFlow + models; one prediction alone pushed memory to ~470MB. Fixed with `GUNICORN_WORKERS=1`, an LRU-capped model cache (`MAX_CACHED_MODELS=2`) in `AssetAwarePredictor`, and periodic worker recycling (`--max-requests`) since TensorFlow's allocator doesn't return freed memory to the OS on its own. Verified prediction output is unchanged (e.g. `RELIANCE.NS` LSTM prediction is byte-for-byte identical before/after).
8. **Vercel's `amondnet/vercel-action@v25` bundles an outdated CLI** that Vercel's backend now rejects. Replaced with installing the Vercel CLI directly in the workflow and calling `vercel deploy` ourselves.
9. **Vercel CLI applies the linked project's Root Directory setting relative to the current working directory.** Running the deploy step from inside `frontend/` doubled it into a nonexistent `frontend/frontend` path; fixed by running from the repo root.
10. **Vercel personal access tokens are scoped at creation time.** A token scoped to a personal account can't read project settings for a project living under a Team — had to recreate the token with the team explicitly selected.

None of these required touching the LSTM/Transformer models, training code, or prediction logic — every fix was configuration, environment handling, or memory management.

## Known gaps carried forward

See [Known limitations](../README.md#known-limitations) in the root README: no `predictor` app tests, no frontend component tests yet, Render free-tier cold starts, and the inherent memory ceiling of running TensorFlow inference on a 512MB host.
