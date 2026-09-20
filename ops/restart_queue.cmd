@echo off
REM Restart the job runner cleanly, then run everything in ops\queue\pending.
REM
REM   ops\restart_queue.cmd
REM
REM Stops whatever runner holds the lock (a runner started before a fix to queue_runner.ps1 is
REM still executing the OLD script -- PowerShell reads a script once, at start), clears a stale
REM lock, and starts a fresh one in this window. Ctrl+C stops it; `New-Item
REM ops\queue\stop -ItemType File` lets it finish the current job first.
setlocal
cd /d D:\bringwrench
echo stopping any running queue runner ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$l='D:\bringwrench\ops\queue\runner.lock'; if (Test-Path $l) { $p=(Get-Content $l -Raw).Trim(); try { Stop-Process -Id ([int]$p) -Force -ErrorAction Stop; Write-Host \"  stopped runner pid $p\" } catch { Write-Host '  no live runner held the lock' }; Remove-Item $l -Force -ErrorAction SilentlyContinue }"
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File D:\bringwrench\ops\queue_runner.ps1
endlocal
