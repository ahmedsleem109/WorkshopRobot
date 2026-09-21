@echo off
REM T7 Tier 1, test 1: replay each demonstration's OWN actions through the serving execution loop,
REM 4 cells (absolute/delta x legs stand-locked/free). Detached, CPU only.
cd /d D:\bringwrench
D:\hexapod\render_venv\Scripts\python.exe scripts\vla_exec_check.py 8 > "C:\Users\LEGION\AppData\Local\Temp\claude\D--bringwrench\ea1c93ce-d255-433f-bffe-4f933e9d438a\scratchpad\exec_check.txt" 2>&1
echo EXEC CHECK DONE >> "C:\Users\LEGION\AppData\Local\Temp\claude\D--bringwrench\ea1c93ce-d255-433f-bffe-4f933e9d438a\scratchpad\exec_check.txt"
