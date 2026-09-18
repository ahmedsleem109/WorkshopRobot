@echo off
wsl.exe -d Ubuntu -- bash -c "tr -d '\r' < /mnt/d/bringwrench/ops/thermal_guard.sh > /tmp/tg.sh && bash /tmp/tg.sh 88"
