@echo off
REM Scheduled-task wrapper: survives the agent session ending.
REM   schtasks /create /tn BwMolmo /tr "D:\bringwrench\ops\run_molmo_download.bat" /sc once /st 23:59 /f
REM   schtasks /run /tn BwMolmo  &&  schtasks /delete /tn BwMolmo /f
wsl.exe -d Ubuntu -- bash -c "tr -d '\r' < /mnt/d/bringwrench/ops/resume_molmo.sh > /tmp/rm.sh && bash /tmp/rm.sh"
