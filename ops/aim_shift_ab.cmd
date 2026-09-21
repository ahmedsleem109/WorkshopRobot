@echo off
REM Session 7: A/B the place AIM SHIFT (the 45 mm anti-topple prior vs none) on table A, 10 seeds x
REM 5 tools per arm, teleported base so this scores the PLACE and not the walk. Detached on purpose
REM -- an agent must not own a long job -- and CPU only, so it runs beside the GPU queue. It WAITS
REM for the noise collection to finish first: run together they starved both (the MJX trainer fell
REM to ~230 steps/s against a measured 2,030).
cd /d D:\bringwrench
for /L %%i in (1,1,240) do (
  findstr /C:"collection done" D:\bw_data\noise_collect.log >nul 2>&1 && goto ready
  timeout /t 30 /nobreak >nul
)
echo the collection never finished -- running the A/B anyway
:ready
set BW_PLACE_AIM_SHIFT=0.045
D:\hexapod\render_venv\Scripts\python.exe scripts\topple_diagnose.py 10 --tool all --table table_a > "C:\Users\LEGION\AppData\Local\Temp\claude\D--bringwrench\ea1c93ce-d255-433f-bffe-4f933e9d438a\scratchpad\ab_shift_045.txt" 2>&1
set BW_PLACE_AIM_SHIFT=0.0
D:\hexapod\render_venv\Scripts\python.exe scripts\topple_diagnose.py 10 --tool all --table table_a > "C:\Users\LEGION\AppData\Local\Temp\claude\D--bringwrench\ea1c93ce-d255-433f-bffe-4f933e9d438a\scratchpad\ab_shift_000.txt" 2>&1
echo AB DONE > "C:\Users\LEGION\AppData\Local\Temp\claude\D--bringwrench\ea1c93ce-d255-433f-bffe-4f933e9d438a\scratchpad\ab_done.txt"
