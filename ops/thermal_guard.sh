#!/bin/bash
# Stop training if the GPU reaches the STATUS.md stop line (>=88 C).
# Exists because `nvidia-smi -lgc` needs an Administrator shell and does NOT survive a
# driver re-init -- the cap lapsed mid-run on 2026-09-18 and the card went back to 1590 MHz.
LIMIT=${1:-88}
LOG=~/bringwrench/logs/thermal_guard.log
echo "$(date +%H:%M:%S) guard armed, limit ${LIMIT}C" >> "$LOG"
while true; do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    if [ -n "$t" ] && [ "$t" -ge "$LIMIT" ]; then
        echo "$(date +%H:%M:%S) TEMP ${t}C >= ${LIMIT}C -- stopping training" >> "$LOG"
        pkill -f 'bw[.]locomotion[.]train_payload'
        echo "$(date +%H:%M:%S) training stopped" >> "$LOG"
        exit 0
    fi
    if ! pgrep -f 'bw[.]locomotion[.]train_payload' > /dev/null; then
        echo "$(date +%H:%M:%S) training gone, guard exiting (last temp ${t}C)" >> "$LOG"
        exit 0
    fi
    sleep 30
done
