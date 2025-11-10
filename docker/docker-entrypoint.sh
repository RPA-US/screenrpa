#!/bin/bash
set -euo pipefail

MARKER=/screenrpa/.initialized

if [ ! -f "$MARKER" ]; then
  echo "[entrypoint] Running one-time initialization..."

  if [ -f /screenrpa/docker/.env ]; then
    cp /screenrpa/docker/.env /screenrpa/core/.env || true
  else 
    echo "[entrypoint] No /screenrpa/docker/.env file found"
    exit 1
  fi

  VENV_PY=/screenrpa/venv/bin/python
  if [ -x "$VENV_PY" ]; then
    PY="$VENV_PY"
  else
    PY=python
  fi

  echo "[entrypoint] Running makemigrations..."
  "$PY" manage.py makemigrations apps_analyzer apps_behaviourmonitoring apps_decisiondiscovery apps_featureextraction apps_processdiscovery apps_reporting || true

  echo "[entrypoint] Running migrate..."
  "$PY" manage.py migrate || true

  echo "[entrypoint] Compiling translations..."
  "$PY" manage.py compilemessages || true

  echo "[entrypoint] Collecting static files..."
  "$PY" manage.py collectstatic --noinput || true

  # Create marker in persistent volume so these steps won't run again
  touch "$MARKER"
  echo "[entrypoint] Initialization complete. Marker created at $MARKER"
fi

exec "$@"
