@echo off
REM T7.3 / T7.4 -- score one SmolVLA checkpoint on the held-out seeds, end to end.
REM
REM   ops\run_eval_vla.bat [CKPT] [N] [OUT]
REM     CKPT  checkpoint dir name under runs/vla_full/checkpoints (default: last)
REM     N     held-out seeds            (default: 20)
REM     OUT   result json               (default: runs\eval\vla_<CKPT>.json)
REM
REM Two processes are needed and this starts both: the policy server runs in WSL on the GPU
REM (port 8766, no auto-start -- unlike the Qwen grounding server on 8765), the evaluator runs
REM in the Windows render venv because EGL/OpenGL does not work inside WSL. The server is
REM stopped again on the way out, so the GPU is free for the next job.
setlocal
set CKPT=%~1
set N=%~2
set OUT=%~3
set ACTSTEPS=%~4
set DELTA=%~5
set RUN=%~6
if "%RUN%"=="" set RUN=vla_full
if "%CKPT%"=="" set CKPT=last
if "%N%"=="" set N=20
if "%OUT%"=="" set OUT=runs\eval\vla_%CKPT%.json
set RENDER_PY=D:\hexapod\render_venv\Scripts\python.exe

echo [1/3] starting the policy server on %CKPT% ...
start "bw-vla-server" wsl.exe -e bash -lc "bash /mnt/d/bringwrench/ops/vla_server.sh ~/bringwrench/runs/%RUN%/checkpoints/%CKPT%/pretrained_model cuda %ACTSTEPS% %DELTA%"

echo [2/3] waiting for it to load (~40 s) ...
set /a TRIES=0
:wait
powershell -NoProfile -Command "try{ Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8766/health -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if not errorlevel 1 goto ready
set /a TRIES+=1
if %TRIES% GEQ 60 (echo    server did not become healthy -- check the bw-vla-server window & goto cleanup)
timeout /t 5 /nobreak >nul
goto wait

:ready
echo [3/3] evaluating %N% held-out seeds -^> %OUT%
"%RENDER_PY%" scripts\eval_vla.py %N% --transfer --swap --out "%OUT%"

:cleanup
REM Stop the server and VERIFY it is gone. A single unchecked pkill is not enough: when it
REM missed once (2026-09-20) the server outlived the job, kept 1.76 GB of VRAM, and the queue
REM wedged behind it. Retry, then confirm the health endpoint has stopped answering.
echo stopping the policy server ...
set /a KILLTRY=0
:kill
wsl.exe -e bash -lc "pkill -f 'vla_server[.]py'" >nul 2>&1
timeout /t 3 /nobreak >nul
powershell -NoProfile -Command "try{ Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8766/health -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 goto killed
set /a KILLTRY+=1
if %KILLTRY% LSS 5 goto kill
echo    WARNING: the policy server is still answering on 8766 -- kill it by hand
:killed
echo done: %OUT%
endlocal
