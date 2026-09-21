@echo off
REM Session 7 regression for the place changes (the pre-release PLACE_MAX_OFFSET guard and the
REM lean-conditional aim shift): the SAME acceptance benchmark the 120/125 and 124/125 numbers came
REM from -- 25 fixed seeds x 5 tools per table, on legs, walking between stations. One process at a
REM time: the host has 15.9 GB and the MJX trainer holds 3.65 GB of it.
cd /d D:\bringwrench
D:\hexapod\render_venv\Scripts\python.exe scripts\try_place.py 25 --table table_a --walk > "C:\Users\LEGION\AppData\Local\Temp\claude\D--bringwrench\ea1c93ce-d255-433f-bffe-4f933e9d438a\scratchpad\place_a.txt" 2>&1
D:\hexapod\render_venv\Scripts\python.exe scripts\try_place.py 25 --table table_b --walk > "C:\Users\LEGION\AppData\Local\Temp\claude\D--bringwrench\ea1c93ce-d255-433f-bffe-4f933e9d438a\scratchpad\place_b.txt" 2>&1
echo PLACE REGRESSION DONE > "C:\Users\LEGION\AppData\Local\Temp\claude\D--bringwrench\ea1c93ce-d255-433f-bffe-4f933e9d438a\scratchpad\place_done.txt"
