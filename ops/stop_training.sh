#!/bin/bash
# Stop the payload run and its two managers, in that order: managers first so the cycle
# manager cannot SIGCONT a process we are about to kill, or log a spurious "training gone".
pkill -f 'tc[.]sh' && echo "cycle manager stopped"
pkill -f 'tg[.]sh' && echo "thermal guard stopped"
sleep 1
PID=$(pgrep -f 'bw[.]locomotion[.]train_payload' | head -1)
if [ -n "$PID" ]; then
    kill -CONT "$PID" 2>/dev/null          # in case it is mid-pause
    pkill -f 'bw[.]locomotion[.]train_payload'
    echo "training (pid $PID) stopped"
else
    echo "training was not running"
fi
sleep 5
echo "--- remaining ---"
pgrep -af 'train_payload|tc[.]sh|tg[.]sh' || echo "all stopped"
echo "--- gpu ---"
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader
echo "--- checkpoints ---"
ls -la ~/bringwrench/runs/results/*payload/checkpoints/ | tail -20
