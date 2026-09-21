@echo off
REM Resume the paused MJX trainer after 30 minutes. Detached on purpose: an agent's background shell
REM gets reaped under memory pressure (four times this session), and a resume that never fires would
REM leave a training run stopped indefinitely. SIGCONT, not a restart -- no progress is lost.
cd /d D:\bringwrench
timeout /t 1800 /nobreak >nul
wsl.exe -e bash -lc "pkill -CONT -f 'bw[.]locomotion[.]train_payload'; sleep 2; pgrep -af 'bw[.]locomotion[.]train_payload' | head -1" > D:\bw_data\resume_pause.log 2>&1
echo RESUMED >> D:\bw_data\resume_pause.log
