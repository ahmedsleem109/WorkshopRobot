@echo off
REM Detached launcher for the T7 noise-injected re-collection (CPU + OpenGL only, no CUDA:
REM it runs alongside a GPU training job on purpose).
cd /d D:\bringwrench
"C:\Program Files\Git\bin\bash.exe" -lc "bash /d/bringwrench/ops/collect_noise.sh 200000 200600 2" > D:\bw_data\noise_collect.log 2>&1
