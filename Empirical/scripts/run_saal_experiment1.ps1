$ErrorActionPreference = "Stop"

$root = "./output_Experiment1/saal"
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

function Run-IfNeeded {
  param(
    [string]$RunName,
    [string[]]$CmdArgs
  )
  $runDir = Join-Path $root $RunName
  if (Test-RunCompleted -RunDir $runDir) {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] SKIP  $RunName (already completed)"
    return
  }
  Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $RunName"
  & python main.py @CmdArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Run failed: $RunName (exit=$LASTEXITCODE)"
  }
  Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DONE  $RunName"
}

New-Item -ItemType Directory -Force -Path $root | Out-Null

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
  "--num-workers", "0",
  "--device", "cuda",
  "--saal-rho", "0.05",
  "--saal-norm", "linf",
  "--output-dir", $root
)

foreach ($seed in 0..9) {
  $pureName = "exp__pure_seed${seed}_i500_b200_r10_pool5000"
  $pureArgs = @($common + @(
    "--acquisition_method", "saal",
    "--seed", "$seed",
    "--run-name", $pureName
  ))
  Run-IfNeeded -RunName $pureName -CmdArgs $pureArgs

  foreach ($p in $percentiles) {
    $pTag = [int]($p * 100)
    $dualName = "exp__dual_b_p${pTag}_seed${seed}_i500_b200_r10_pool5000"
    $dualArgs = @($common + @(
      "--acquisition_method", "saal_dual_b",
      "--dual-percentile", "$p",
      "--seed", "$seed",
      "--run-name", $dualName
    ))
    Run-IfNeeded -RunName $dualName -CmdArgs $dualArgs
  }
}

Write-Host "SAAL Experiment1 runs completed."
