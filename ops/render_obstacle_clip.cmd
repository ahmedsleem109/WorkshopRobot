@echo off
REM T12: the one session-6 result the montage does not show yet -- the obstacle detour (0/2 ->
REM 10/10). Detached and CPU-only, so it runs beside the GPU queue. The command text comes from
REM eval_suite's own obstacle scenario, so only the seed is given here.
cd /d D:\bringwrench
D:\hexapod\render_venv\Scripts\python.exe scripts\make_orch_video.py 0 media\recover_obstacle.mp4 --suite obstacle > D:\bw_data\render_obstacle.log 2>&1
echo RENDER DONE >> D:\bw_data\render_obstacle.log
