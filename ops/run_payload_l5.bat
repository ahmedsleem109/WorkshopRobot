@echo off
wsl.exe -d Ubuntu -- bash -c "tr -d '\r' < /mnt/d/bringwrench/ops/train_payload_l5.sh > /tmp/tp5.sh && bash /tmp/tp5.sh"
