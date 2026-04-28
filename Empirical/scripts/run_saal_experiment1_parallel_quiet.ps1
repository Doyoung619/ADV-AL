param(
  [int]$MaxParallel = 2,
  [int]$NumWorkers = 1,
  [switch]$SkipCompleted = $true
)

$ErrorActionPreference = "Stop"

$root = "./output_Experiment1/saal"
$logRoot = Join-Path $root "_launcher_logs"
New-Item -ItemType Directory -Force -Path $root | Out-Null
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$eps = "0.0039215686"  # 1/255
$percentiles = @(0.1, 0.3, 0.5, 0.7, 0.9)

function Test-RunCompleted {
  param([string]$RunDir)
  $metrics = Join-Path $RunDir "round_metrics.json"
  if (!(Test-Path $metrics)) { return $false }
  try {
    $rows = Get-Content $metrics -Raw | ConvertFrom-Json
    if ($null -ne $rows -and $rows.Count -ge 11) { return $true }
  } catch {}
  return $false
}

$common = @(
  "--dataset", "cifar10",
  "--model", "small_cnn",
  "--initial_labeled_size", "500",
  "--acquisition_batch_size", "200",
  "--rounds", "10",
  "--epochs_per_round", "50",
  "--epsilon", $eps,
  "--epsilon-acq", $eps,
  "--acquisition-pool-subset-size", "5000",
  "--skip-logit-mismatch-eval",
  "--num-workers", "$NumWorkers",
  "--device", "cuda",
  "--saal-rho", "0.05",
  "--saal-norm", "linf",
  "--output-dir", $root
)

$jobs = New-Object System.Collections.Generic.List[object]
$skippedCompleted = 0

foreach ($seed in 0..9) {
  $pureName = "exp__pure_seed${seed}_i500_b200_r10_pool5000"
  $pureDir = Join-Path $root $pureName
  if ($SkipCompleted -and (Test-RunCompleted -RunDir $pureDir)) {
    $skippedCompleted += 1
  } else {
    $args = @($common + @(
      "--acquisition_method", "saal",
      "--seed", "$seed",
      "--run-name", $pureName
    ))
    $jobs.Add([pscustomobject]@{ run_name=$pureName; method="saal"; args=$args })
  }

  foreach ($p in $percentiles) {
    $pTag = [int]($p * 100)
    $dualName = "exp__dual_b_p${pTag}_seed${seed}_i500_b200_r10_pool5000"
    $dualDir = Join-Path $root $dualName
    if ($SkipCompleted -and (Test-RunCompleted -RunDir $dualDir)) {
      $skippedCompleted += 1
    } else {
      $args = @($common + @(
        "--acquisition_method", "saal_dual_b",
        "--dual-percentile", "$p",
        "--seed", "$seed",
        "--run-name", $dualName
      ))
      $jobs.Add([pscustomobject]@{ run_name=$dualName; method="saal_dual_b"; args=$args })
    }
  }
}

Write-Host "Remaining runs: $($jobs.Count)"
Write-Host "Skipped completed runs: $skippedCompleted"
Write-Host "Max parallel: $MaxParallel | num-workers per run: $NumWorkers | device: cuda"

$running = New-Object System.Collections.Generic.List[object]
$cursor = 0
$failed = New-Object System.Collections.Generic.List[object]

while ($cursor -lt $jobs.Count -or $running.Count -gt 0) {
  while ($cursor -lt $jobs.Count -and $running.Count -lt $MaxParallel) {
    $j = $jobs[$cursor]
    $cursor += 1

    $stdoutPath = Join-Path $logRoot "$($j.run_name).stdout.log"
    $stderrPath = Join-Path $logRoot "$($j.run_name).stderr.log"

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $($j.run_name)"
    $argList = @("main.py") + $j.args
    $proc = Start-Process -FilePath "python" -ArgumentList $argList -NoNewWindow -PassThru `
      -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

    $running.Add([pscustomobject]@{
      meta = $j
      proc = $proc
      stdout = $stdoutPath
      stderr = $stderrPath
      started_at = Get-Date
    })
  }

  Start-Sleep -Seconds 5

  for ($i = $running.Count - 1; $i -ge 0; $i--) {
    $r = $running[$i]
    if ($r.proc.HasExited) {
      $elapsed = (Get-Date) - $r.started_at
      if ($r.proc.ExitCode -eq 0) {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DONE  $($r.meta.run_name) (${elapsed})"
      } else {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] FAIL  $($r.meta.run_name) exit=$($r.proc.ExitCode)"
        $failed.Add([pscustomobject]@{
          run_name = $r.meta.run_name
          method = $r.meta.method
          exit_code = $r.proc.ExitCode
          stdout = $r.stdout
          stderr = $r.stderr
        })
      }
      $running.RemoveAt($i)
    }
  }
}

if ($failed.Count -gt 0) {
  $failedPath = Join-Path $logRoot "failed_runs_saal.json"
  $failed | ConvertTo-Json -Depth 4 | Out-File -Encoding utf8 $failedPath
  throw "Completed with failures: $($failed.Count). See $failedPath"
}

Write-Host "All SAAL Experiment1 runs completed."

