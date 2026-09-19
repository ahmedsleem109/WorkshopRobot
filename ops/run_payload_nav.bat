@echo off
wsl.exe -d Ubuntu -- bash -c "tr -d '\r' < /mnt/d/bringwrench/ops/train_payload_nav.sh > /tmp/tpn.sh && bash /tmp/tpn.sh"
