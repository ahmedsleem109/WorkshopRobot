@echo off
wsl.exe -d Ubuntu -- bash -c "tr -d '\r' < /mnt/d/bringwrench/ops/train_payload_fg.sh > /tmp/tp.sh && bash /tmp/tp.sh"
