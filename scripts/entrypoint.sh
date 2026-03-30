#!/bin/bash
# entrypoint.sh
#
# Container entrypoint:
#   1. Run model export / library compile (idempotent)
#   2. Launch the DeepStream + NetworkTables probe application

set -euo pipefail

echo "============================================"
echo " DeepStream YOLOv11n + NetworkTables Client"
echo " Team: ${NT_TEAM_NUMBER:-<not set>}"
echo "============================================"

/opt/deepstream/scripts/export_model.sh

echo "[entrypoint] Starting ds_nt_probe..."
exec /opt/deepstream/probe_app/build/ds_nt_probe
