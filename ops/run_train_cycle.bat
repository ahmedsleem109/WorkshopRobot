@echo off
wsl.exe -d Ubuntu -- bash -c "tr -d '\r' < /mnt/d/bringwrench/ops/train_cycle.sh > /tmp/tc.sh && bash /tmp/tc.sh 10800 900"
