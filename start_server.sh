#!/bin/bash
pkill -f "triune.api" || true
pkill -f "uvicorn" || true
sleep 1
LOG_FILE="/home/yasho4867/triune_server.log"
nohup /home/yasho4867/venvs/triune/bin/python -m uvicorn triune.api.server:create_app --factory --host 0.0.0.0 --port 8000 --app-dir /mnt/c/Users/yashb_f1ls/OneDrive/Documents/TriuneTransformer > "$LOG_FILE" 2>&1 &
PID=$!
sleep 3
echo "Triune Server started with PID: $PID"
