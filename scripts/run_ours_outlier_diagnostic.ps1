param(
  [int]$Seed = 0,
  [string]$Device = "cuda",
  [string]$UncertaintyMethod = "bald",
  [string]$OursMethod = "ours_l2",
  [double]$FilterPercentile = 10,
  [int]$CandidatePoolSize = 5000,
  [int]$NumWorkers = 0,
  [double]$Epsilon = 0.0313725490,
  [string]$AcquisitionAttack = "pgd",
  [int]$AcquisitionPgdSteps = 5,
  [double]$AcquisitionPgdAlpha = 0.0078431373
)

$ErrorActionPreference = "Stop"

$cmd = @(
  "analysis/analyze_ours_outlierness.py",
  "--dataset", "cifar10",
  "--model", "small_cnn",
  "--initial-labeled-size", "500",
  "--acquisition-size", "200",
  "--uncertainty-method", $UncertaintyMethod,
  "--filter-percentile", "$FilterPercentile",
  "--analysis-candidate-pool-size", "$CandidatePoolSize",
  "--outlier-k", "10",
  "--ours-method", $OursMethod,
  "--seed", "$Seed",
  "--output-dir", "./output_Experiment2",
  "--epsilon", "$Epsilon",
  "--acquisition-attack", $AcquisitionAttack,
  "--acquisition-pgd-steps", "$AcquisitionPgdSteps",
  "--acquisition-pgd-alpha", "$AcquisitionPgdAlpha",
  "--device", $Device,
  "--num-workers", "$NumWorkers"
)

Write-Host "Running diagnostic analysis..."
Write-Host "python $($cmd -join ' ')"
& python @cmd
if ($LASTEXITCODE -ne 0) {
  throw "Diagnostic analysis failed (exit=$LASTEXITCODE)."
}
Write-Host "Done."
