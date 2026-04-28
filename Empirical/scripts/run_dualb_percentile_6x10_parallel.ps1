param(
  [int]$MaxParallel = 2,
  [int]$BaldMaxParallel = 1,
  [int]$NumWorkers = 2,
  [string]$Device = "cuda",
  [switch]$NoDeterministic,
  [switch]$SkipCompleted = $true,
  [string[]]$Algorithms = @("badge", "bald", "entropy")
)

$ErrorActionPreference = "Stop"

# 3 algorithms x (dual_b percentiles 10/30/50/70/90 + pure) x 10 seeds = 180 runs
$algorithms = @(
  @{ name = "badge";   pure = "badge";   dual = "badge_dual_b" },
  @{ name = "bald";    pure = "bald";    dual = "bald_dual_b" },
  @{ name = "entropy"; pure = "entropy"; dual = "entropy_dual_b" }
)
$percentiles = @(0.1, 0.3, 0.5, 0.7, 0.9)
$seeds = 0..9

$root = "./output_Experiment1"
New-Item -ItemType Directory -Force -Path $root | Out-Null
$launcherLogRoot = Join-Path $root "_launcher_logs"
New-Item -ItemType Directory -Force -Path $launcherLogRoot | Out-Null

$baseCommonArgs = @(
  "--dataset", "cifar10",
  "--model", "small_cnn",
  "--initial_labeled_size", "500",
  "--acquisition_batch_size", "200",
  "--rounds", "10",
  "--epochs_per_round", "50",
  "--epsilon", "0.0039215686",
  "--epsilon-acq", "0.0039215686",
  "--acquisition-pool-subset-size", "5000",
  "--skip-logit-mismatch-eval",
  "--num-workers", "$NumWorkers",
  "--device", $Device
)
if ($NoDeterministic) {
  $baseCommonArgs += "--no-deterministic"
}

# Build full job list first.
$jobs = New-Object System.Collections.Generic.List[object]
$skippedCompleted = 0
foreach ($algo in $algorithms) {
  if ($Algorithms -notcontains $algo.name) {
    continue
  }
  $algoOutDir = Join-Path $root $algo.name
  New-Item -ItemType Directory -Force -Path $algoOutDir | Out-Null
  $algoLogDir = Join-Path $launcherLogRoot $algo.name
  New-Item -ItemType Directory -Force -Path $algoLogDir | Out-Null

  foreach ($seed in $seeds) {
    $pureRunName = "exp_$($algo.name)_pure_seed${seed}_i500_b200_r10_pool5000"
    $pureRunDir = Join-Path $algoOutDir $pureRunName
    $pureRoundMetrics = Join-Path $pureRunDir "round_metrics.json"
    # Backward-compat: older run-name format "exp__..."
    $legacyPureRunName = "exp__pure_seed${seed}_i500_b200_r10_pool5000"
    $legacyPureRunDir = Join-Path $algoOutDir $legacyPureRunName
    $legacyPureRoundMetrics = Join-Path $legacyPureRunDir "round_metrics.json"
    $pureCompleted = $false
    if ($SkipCompleted) {
      foreach ($cand in @($pureRoundMetrics, $legacyPureRoundMetrics)) {
        if (Test-Path $cand) {
          try {
            $rr = Get-Content $cand -Raw | ConvertFrom-Json
            if ($null -ne $rr -and $rr.Count -ge 11) {
              $pureCompleted = $true
              break
            }
          } catch {}
        }
      }
    }
    if ($pureCompleted) {
      $skippedCompleted += 1
    } else {
      $jobs.Add([pscustomobject]@{
      run_name = $pureRunName
      algo_name = $algo.name
      method = $algo.pure
      seed = $seed
      dual_percentile = $null
      })
    }

    foreach ($p in $percentiles) {
      $pTag = [int]($p * 100)
      $dualRunName = "exp_$($algo.name)_dual_b_p${pTag}_seed${seed}_i500_b200_r10_pool5000"
      $dualRunDir = Join-Path $algoOutDir $dualRunName
      $dualRoundMetrics = Join-Path $dualRunDir "round_metrics.json"
      # Backward-compat: older run-name format "exp__..."
      $legacyDualRunName = "exp__dual_b_p${pTag}_seed${seed}_i500_b200_r10_pool5000"
      $legacyDualRunDir = Join-Path $algoOutDir $legacyDualRunName
      $legacyDualRoundMetrics = Join-Path $legacyDualRunDir "round_metrics.json"
      $dualCompleted = $false
      if ($SkipCompleted) {
        foreach ($cand in @($dualRoundMetrics, $legacyDualRoundMetrics)) {
          if (Test-Path $cand) {
            try {
              $rr = Get-Content $cand -Raw | ConvertFrom-Json
              if ($null -ne $rr -and $rr.Count -ge 11) {
                $dualCompleted = $true
                break
              }
            } catch {}
          }
        }
      }
      if ($dualCompleted) {
        $skippedCompleted += 1
      } else {
        $jobs.Add([pscustomobject]@{
        run_name = $dualRunName
        algo_name = $algo.name
        method = $algo.dual
        seed = $seed
        dual_percentile = $p
        })
      }
    }
  }
}

Write-Host "Total runs: $($jobs.Count)"
Write-Host "Max parallel: $MaxParallel | num-workers per run: $NumWorkers | device: $Device"
Write-Host "Bald max parallel: $BaldMaxParallel"
Write-Host "Skipped already completed runs: $skippedCompleted"

$running = New-Object System.Collections.Generic.List[object]
$cursor = 0
$failed = New-Object System.Collections.Generic.List[object]

while ($cursor -lt $jobs.Count -or $running.Count -gt 0) {
  # Launch new tasks while slots are free.
  while ($cursor -lt $jobs.Count -and $running.Count -lt $MaxParallel) {
    $j = $jobs[$cursor]
    if ($j.algo_name -eq "bald") {
      $runningBald = @($running | Where-Object { $_.meta.algo_name -eq "bald" }).Count
      if ($runningBald -ge $BaldMaxParallel) {
        break
      }
    }
    $cursor += 1

    $args = New-Object System.Collections.Generic.List[string]
    $args.Add("main.py")
    foreach ($a in $baseCommonArgs) { $args.Add($a) }
    $algoOutDir = Join-Path $root $j.algo_name
    New-Item -ItemType Directory -Force -Path $algoOutDir | Out-Null
    $args.Add("--output-dir"); $args.Add($algoOutDir)
    $args.Add("--acquisition_method"); $args.Add($j.method)
    if ($null -ne $j.dual_percentile) {
      $args.Add("--dual-percentile"); $args.Add([string]$j.dual_percentile)
    }
    $args.Add("--seed"); $args.Add([string]$j.seed)
    $args.Add("--run-name"); $args.Add($j.run_name)

    $jobAlgoLogDir = Join-Path $launcherLogRoot $j.algo_name
    New-Item -ItemType Directory -Force -Path $jobAlgoLogDir | Out-Null
    $stdoutPath = Join-Path $jobAlgoLogDir "$($j.run_name).stdout.log"
    $stderrPath = Join-Path $jobAlgoLogDir "$($j.run_name).stderr.log"

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $($j.run_name)"
    $proc = Start-Process -FilePath "python" -ArgumentList $args -NoNewWindow -PassThru `
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

  # Collect finished tasks.
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
          seed = $r.meta.seed
          dual_percentile = $r.meta.dual_percentile
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
  $failedPath = Join-Path $root "failed_runs.json"
  $failed | ConvertTo-Json -Depth 4 | Out-File -Encoding utf8 $failedPath
  throw "Completed with failures: $($failed.Count). See $failedPath"
}

Write-Host "All dual_b percentile experiments completed successfully."
