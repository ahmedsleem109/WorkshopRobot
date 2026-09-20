@echo off
REM Register the queue runner as a Windows Scheduled Task, so it survives closing the terminal
REM and logging out -- the fully detached option in STATUS.md's "Long jobs" note.
REM
REM   ops\install_queue_task.cmd            register and start it now
REM   ops\install_queue_task.cmd remove     unregister it
REM
REM This is a USER-level task: no Administrator shell needed. Unlike the one-shot BwTrain task
REM in STATUS.md, this one is MEANT to persist -- it re-fires at logon and then just watches an
REM empty queue, which costs nothing.
setlocal
set TASK=BwQueue
set PS=powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File D:\bringwrench\ops\queue_runner.ps1

if /I "%~1"=="remove" (
  schtasks /delete /tn %TASK% /f
  echo removed %TASK%
  goto :eof
)

schtasks /create /tn %TASK% /tr "%PS%" /sc onlogon /rl limited /f
if errorlevel 1 (echo could not register %TASK% & goto :eof)
schtasks /run /tn %TASK%
echo.
echo %TASK% registered and started. It runs hidden -- watch it with:
echo   type D:\bringwrench\runs\queue\status.json
echo   schtasks /query /tn %TASK%
endlocal
