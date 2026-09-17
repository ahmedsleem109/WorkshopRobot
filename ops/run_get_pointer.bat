@echo off
wsl.exe -d Ubuntu -- bash -c "tr -d '\r' < /mnt/d/bringwrench/ops/get_pointer.sh > /tmp/gp.sh && bash /tmp/gp.sh"
