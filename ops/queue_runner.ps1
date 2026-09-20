<#
    A detached job runner for this project's long GPU jobs.

        powershell -NoProfile -ExecutionPolicy Bypass -File ops\queue_runner.ps1

    WHY THIS EXISTS (session 6, 2026-09-20). A long job launched from an agent's background
    shell dies with that shell: Claude Code reaps its own background shells under memory
    pressure, and it killed the same SmolVLA fine-tune twice, at steps 3,900 and 4,700 of
    6,000. The same run, launched from a plain terminal, finished untouched. So the rule is:
    the agent must never OWN a long job. It enqueues one by WRITING A FILE -- which nothing
    can reap -- and this runner, started by a human once, executes it.

    It also enforces the standing rules that a human otherwise has to remember:
      * ONE GPU JOB AT A TIME. The 6 GB card cannot hold two, and stacking them is what the
        project's own notes forbid. Jobs run strictly one after another -- AND a job waits
        while ANY other process is on the card, so it is safe to start this runner while a
        hand-launched training run is still going. It will simply queue behind it.
      * COOL THE CARD FIRST. A job waits while the GPU is at or above -MaxGpuTemp (88 C is
        the project's stop threshold).

    LAYOUT (all under ops/queue/):
        pending/    jobs waiting, run in FILENAME ORDER -- name them 010-train.cmd, 020-eval.cmd
        running/    the one job in flight
        done/       finished, exit code 0
        failed/     finished, non-zero exit
        stop        create this file to make the runner exit after the current job

    Each job is an ordinary .cmd file run by cmd.exe from the repo root. Its combined output
    goes to runs/queue/<job>.log, and runs/queue/status.json always says what is running, what
    finished and with what exit code -- so the agent can read progress without owning anything.
#>
param(
    [string]$Root = "D:\bringwrench",
    [int]$PollSeconds = 10,
    [int]$MaxGpuTemp = 88,
    [int]$FreeGpuBelowMB = 1500,   # above this, something else owns the card -- wait for it
    [switch]$Once,
    # --- the messenger half (OFF by default; read the security note below before using it)
    [switch]$NotifyClaude,
    [string]$ClaudeExe = "$env:APPDATA\npm\claude",
    [string]$ClaudeModel = ""
)

$ErrorActionPreference = "Stop"
$QueueDir = Join-Path $Root "ops\queue"
$LogDir = Join-Path $Root "runs\queue"
$Dirs = @("pending", "running", "done", "failed") | ForEach-Object { Join-Path $QueueDir $_ }
foreach ($d in @($Dirs + $LogDir)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}
$StatusPath = Join-Path $LogDir "status.json"
$LockPath = Join-Path $QueueDir "runner.lock"

# --- single instance. A stale lock from a killed runner is taken over, not obeyed.
if (Test-Path $LockPath) {
    $old = (Get-Content $LockPath -Raw).Trim()
    $alive = $null
    try { $alive = Get-Process -Id ([int]$old) -ErrorAction Stop } catch { $alive = $null }
    if ($alive -and $alive.ProcessName -like "*powershell*") {
        Write-Host "another runner is already live (pid $old) -- nothing to do"
        exit 0
    }
    Write-Host "taking over a stale lock from pid $old"
}
Set-Content -Path $LockPath -Value $PID -Encoding utf8

# Windows PowerShell 5.1's -Encoding utf8 means "UTF-8 WITH a BOM", and Tee-Object has no
# -Encoding at all and defaults to UTF-16. Both make these files unreadable to the ordinary
# tools that consume them (json.load chokes on a BOM, grep sees NUL-separated bytes), so
# every write here goes through .NET with an explicit BOM-less UTF-8 encoder.
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Write-TextNoBom([string]$Path, [string]$Text, [bool]$Append = $false) {
    if ($Append) { [System.IO.File]::AppendAllText($Path, $Text, $Utf8NoBom) }
    else { [System.IO.File]::WriteAllText($Path, $Text, $Utf8NoBom) }
}

function Write-Status([hashtable]$State) {
    $State["updated"] = (Get-Date).ToString("s")
    try { Write-TextNoBom $StatusPath (($State | ConvertTo-Json -Depth 4) + "`r`n") } catch { }
}

function Get-GpuTemp {
    try {
        $t = & nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>$null
        if ($t) { return [int]($t | Select-Object -First 1) }
    } catch { }
    return -1          # no nvidia-smi: do not block on a reading we cannot take
}

function Get-GpuMemMB {
    try {
        $m = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
        if ($m) { return [int]($m | Select-Object -First 1) }
    } catch { }
    return 0           # no reading: do not block
}

function Wait-ForGpu {
    # Both gates in one wait: the card must be COOL, and it must be FREE. The second is what
    # makes it safe to leave this runner up while a training run launched by hand is on the
    # card -- the queued job simply waits its turn instead of stacking on 6 GB of VRAM.
    while ($true) {
        $t = Get-GpuTemp
        $m = Get-GpuMemMB
        if ($t -ge $MaxGpuTemp) {
            Write-Host ("  GPU at {0} C (>= {1}) -- waiting 60 s" -f $t, $MaxGpuTemp)
            Write-Status @{ state = "cooling"; gpu_c = $t }
            Start-Sleep -Seconds 60
            continue
        }
        if ($m -gt $FreeGpuBelowMB) {
            Write-Host ("  GPU busy: {0} MiB in use (> {1}) -- waiting 60 s" -f $m, $FreeGpuBelowMB)
            Write-Status @{ state = "waiting_for_gpu"; gpu_mb = $m; gpu_c = $t }
            Start-Sleep -Seconds 60
            continue
        }
        return $t
    }
}

Write-Host "queue runner up (pid $PID). watching $QueueDir\pending"
Write-Host "  stop it cleanly with:  New-Item $QueueDir\stop -ItemType File"
Write-Status @{ state = "idle"; pid = $PID }

while ($true) {
    if (Test-Path (Join-Path $QueueDir "stop")) {
        Write-Host "stop file present -- exiting"
        Remove-Item (Join-Path $QueueDir "stop") -Force -ErrorAction SilentlyContinue
        break
    }

    $job = Get-ChildItem (Join-Path $QueueDir "pending") -Filter *.cmd -ErrorAction SilentlyContinue |
           Sort-Object Name | Select-Object -First 1
    if (-not $job) {
        if ($Once) { Write-Host "queue empty -- exiting (-Once)"; break }
        Write-Status @{ state = "idle"; pid = $PID }
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    $name = $job.BaseName
    $running = Join-Path $QueueDir "running\$($job.Name)"
    Move-Item $job.FullName $running -Force
    $log = Join-Path $LogDir "$name.log"

    # A job that declares `REM NOGPU` skips the GPU gate. Without this a CPU-only job queues
    # behind the card: measured 2026-09-20, the dataset conversion sat in waiting_for_gpu
    # because the grounding server from the PREVIOUS job was still resident with 4.9 GB --
    # `vlm.ensure_server()` starts that server and nothing ever shuts it down.
    $needsGpu = -not (Select-String -Path $running -Pattern '^\s*REM\s+NOGPU' -Quiet)
    if ($needsGpu) { $temp = Wait-ForGpu } else { $temp = Get-GpuTemp }

    $started = Get-Date
    Write-Host ("[{0}] running {1} (GPU {2} C) -> {3}" -f $started.ToString("HH:mm:ss"), $name, $temp, $log)
    Write-Status @{ state = "running"; job = $name; started = $started.ToString("s"); log = $log; pid = $PID }

    $code = 0
    $errLog = "$log.err"
    # Wait on the DIRECT CHILD ONLY. Two ways of running a job both hang when it leaves a
    # process behind -- and these jobs do, because they start a policy server in its own
    # window:
    #   * `& cmd.exe ... | ForEach-Object` reads until every holder of the inherited stdout
    #     handle closes it, grandchildren included;
    #   * `Start-Process -Wait` waits for the process AND ITS DESCENDANTS, by documented design.
    # Measured 2026-09-20: job 020's evaluator finished and wrote its result, but the server it
    # had started outlived it and the queue wedged for 42 minutes behind a job that was done;
    # a job spawning a 25 s grandchild then held -Wait for the full 25 s too.
    # $proc.WaitForExit() waits on that one process and nothing else.
    try {
        $proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "`"$running`"" `
            -WorkingDirectory $Root -RedirectStandardOutput $log -RedirectStandardError $errLog `
            -NoNewWindow -PassThru
        $proc.WaitForExit()
        $code = $proc.ExitCode
        if ($null -eq $code) { $code = 0 }
    } catch {
        Write-TextNoBom $errLog (($_ | Out-String) + "`r`n") $true
        $code = 1
    }
    # Fold stderr into the job log so one file holds the whole story, then show the tail.
    if ((Test-Path $errLog) -and (Get-Item $errLog).Length -gt 0) {
        Write-TextNoBom $log ("`r`n--- stderr ---`r`n" + (Get-Content $errLog -Raw)) $true
    }
    Remove-Item $errLog -Force -ErrorAction SilentlyContinue
    if (Test-Path $log) { Get-Content $log -Tail 12 | ForEach-Object { Write-Host "  $_" } }

    $mins = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
    $dest = "done"
    if ($code -ne 0) { $dest = "failed" }
    Move-Item $running (Join-Path $QueueDir "$dest\$($job.Name)") -Force
    Write-Host ("[{0}] {1} -> {2} (exit {3}, {4} min)" -f (Get-Date).ToString("HH:mm:ss"), $name, $dest, $code, $mins)
    Write-Status @{ state = "finished"; job = $name; result = $dest; exit_code = $code;
                    minutes = $mins; log = $log; pid = $PID }
    # A one-line-per-job history, because status.json only ever holds the latest event. This
    # file is also the MESSENGER channel for a live agent session: it tails this one file and
    # every finished job arrives as a notification it can judge.
    Write-TextNoBom (Join-Path $LogDir "history.log") `
        ("{0}  {1,-28} {2,-6} exit={3} {4} min`r`n" -f (Get-Date).ToString("s"), $name, $dest, $code, $mins) $true

    # --- the messenger, for when NO agent session is live (overnight, or you are away).
    # Hands the finished job to a headless `claude -p`, which reads the log and writes a
    # verdict next to it. SECURITY: this runs an agent UNATTENDED with edit permission, so it
    # can write files in this repo without anyone approving each one. That is the whole point
    # and also the risk -- it stays off unless -NotifyClaude is passed, and it is told not to
    # start anything long itself, so the worst case is a wrong verdict file or a queued job,
    # not a runaway GPU run.
    if ($NotifyClaude) {
        $verdict = Join-Path $LogDir "$name.verdict.md"
        $ask = @"
A queued job just finished in D:\bringwrench (this is an automated hand-off, no human is watching).
    job=$name  result=$dest  exit=$code  minutes=$mins
Read runs/queue/$name.log, plus any result json it names under runs/eval/. Judge whether the
numbers are good, suspicious, or a failure, and say why in at most 8 lines. Compare against
STATUS.md if it records a prior number for the same thing. If an obvious, cheap follow-up is
warranted, write it as a new job file into ops/queue/pending/ (filename ordering decides when it
runs). Do NOT start any long job yourself and do NOT edit STATUS.md.
"@
        $args = @("-p", $ask, "--permission-mode", "acceptEdits")
        if ($ClaudeModel) { $args += @("--model", $ClaudeModel) }
        Write-Host "  handing $name to claude -p ..."
        try {
            & $ClaudeExe @args *>&1 | ForEach-Object {
                Write-TextNoBom $verdict ("$_" + "`r`n") $true
            }
            Write-Host "  verdict -> $verdict"
        } catch {
            Write-TextNoBom $verdict (($_ | Out-String) + "`r`n") $true
            Write-Host "  claude -p failed (see $verdict)"
        }
    }
}

Remove-Item $LockPath -Force -ErrorAction SilentlyContinue
Write-Status @{ state = "stopped" }
