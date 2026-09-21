@echo off
REM Bring the pipeline back after the 30-minute break: clear the queue's stop flag and start the
REM runner again. It picks up 428-resume_nav_l5 first (warm start from step 3,686,400), then the
REM export, the policy swap + seven suites, the grounding retry, and the corrected VLA run.
REM Detached on purpose -- this must not depend on an agent's shell surviving.
cd /d D:\bringwrench
timeout /t 1320 /nobreak >nul
del /Q ops\queue\stop 2>nul
powershell -NoProfile -ExecutionPolicy Bypass -File ops\queue_runner.ps1 > D:\bw_data\runner_restart.log 2>&1
