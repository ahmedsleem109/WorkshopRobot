#!/bin/bash
# Sample GPU temperature/util/memory every 15 s to ~/bringwrench/logs/gputemp.log.
# A script file, not an inline nohup one-liner: nested quoting expands $(...) once at launch
# and the loop then echoes a frozen string (measured 2026-09-18).
OUT=~/bringwrench/logs/gputemp.log
: > "$OUT"
while true; do
    printf '%s %s\n' "$(date +%H:%M:%S)" \
        "$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader)" >> "$OUT"
    sleep 15
done
