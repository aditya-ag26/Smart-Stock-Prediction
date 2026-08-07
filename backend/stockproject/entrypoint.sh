#!/bin/sh
set -e

python manage.py migrate --noinput

# TensorFlow's allocator doesn't return freed memory to the OS even after a
# cached model is evicted, so RSS only ever climbs within a worker's
# lifetime. --max-requests periodically recycles the worker (a fresh
# process actually gets the memory back), trading a brief cold-reload for
# bounded long-term memory on memory-constrained hosts.
exec gunicorn stockproject.wsgi:application \
    --bind "0.0.0.0:${PORT:-8000}" \
    --workers "${GUNICORN_WORKERS:-1}" \
    --timeout 120 \
    --max-requests "${GUNICORN_MAX_REQUESTS:-30}" \
    --max-requests-jitter "${GUNICORN_MAX_REQUESTS_JITTER:-10}"
