#!/bin/bash
# Every WORK_S of training: pause the run, let the GPU cool for REST_S, resume, and append a
# metrics snapshot. Requested 2026-09-18: "check metrics every 3 hours, cool down gpu for 15
# minutes then proceed".
#
# Pause is SIGSTOP/SIGCONT on the live process, NOT a restart: a restart would throw away all
# progress since the last checkpoint and pay the ~10 min XLA compile again. The process keeps
# its CUDA context while stopped; the GPU drains its queue and idles.
#
# This is ON TOP of the in-training duty cycle (rest_every_s 900 / rest_seconds 120), which
# keeps the card near 80 C minute to minute. This adds the deep rest every few hours.
WORK_S=${1:-10800}          # 3 h of training
REST_S=${2:-900}            # 15 min cooling
PAT='bw[.]locomotion[.]train_payload'
LOG=~/bringwrench/logs/train_cycle.log
EVALS=~/bringwrench/logs/${3:-payload}.log

say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }

snapshot() {
    say "--- metrics snapshot ---"
    grep -E 'reward' "$EVALS" | grep -v 'reward      0.00' | tail -4 \
        | cut -c1-150 | sed 's/^/    /' >> "$LOG"
    say "    gpu: $(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,clocks.gr --format=csv,noheader)"
}

say "cycle manager armed: work ${WORK_S}s, rest ${REST_S}s"
while true; do
    # work phase -- bail out early if training ends
    waited=0
    while [ "$waited" -lt "$WORK_S" ]; do
        pgrep -f "$PAT" > /dev/null || { say "training gone; cycle manager exiting"; snapshot; exit 0; }
        sleep 30
        waited=$((waited + 30))
    done

    PID=$(pgrep -f "$PAT" | head -1)
    [ -z "$PID" ] && { say "training gone; exiting"; exit 0; }

    t_before=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    kill -STOP "$PID" && say "PAUSED pid $PID at ${t_before}C -- cooling ${REST_S}s"
    snapshot
    sleep "$REST_S"
    t_after=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
    kill -CONT "$PID" && say "RESUMED pid $PID -- cooled ${t_before}C -> ${t_after}C"
done
