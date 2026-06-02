$ErrorActionPreference = "Stop"

# 3 algorithms x (dual_b percentiles 10/30/50/70/90 + pure) x 10 seeds = 180 runs
$algorithms = @(
  @{ name = "badge";   pure = "badge";   dual = "badge_dual_b" },
  @{ name = "bald";    pure = "bald";    dual = "bald_dual_b" },
  @{ name = "entropy"; pure = "entropy"; dual = "entropy_dual_b" }
)
$percentiles = @(0.1, 0.3, 0.5, 0.7, 0.9)
$seeds = 0..9

$root = "./output_dualb_percentile_6x10"
New-Item -ItemType Directory -Force -Path $root | Out-Null

$commonArgs = @(
  "--dataset", "cifar10",
  "--model", "small_cnn",
  "--initial_labeled_size", "500",
  "--acquisition_batch_size", "200",
  "--rounds", "10",
  "--epochs_per_round", "50",
  "--acquisition-pool-subset-size", "5000",
  "--skip-logit-mismatch-eval",
  "--output-dir", $root
)

foreach ($algo in $algorithms) {
  foreach ($seed in $seeds) {
    $pureRunName = "exp_${($algo.name)}_pure_seed${seed}_i500_b200_r10_pool5000"
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $pureRunName"
    & python main.py @commonArgs `
      --acquisition_method $algo.pure `
      --seed $seed `
      --run-name $pureRunName
    if ($LASTEXITCODE -ne 0) {
      throw "Run failed: $pureRunName (exit=$LASTEXITCODE)"
    }
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DONE  $pureRunName"

    foreach ($p in $percentiles) {
      $pTag = [int]($p * 100)
      $dualRunName = "exp_${($algo.name)}_dual_b_p${pTag}_seed${seed}_i500_b200_r10_pool5000"
      Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $dualRunName"
      & python main.py @commonArgs `
        --acquisition_method $algo.dual `
        --dual-percentile $p `
        --seed $seed `
        --run-name $dualRunName
      if ($LASTEXITCODE -ne 0) {
        throw "Run failed: $dualRunName (exit=$LASTEXITCODE)"
      }
      Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DONE  $dualRunName"
    }
  }
}

Write-Host "All dual_b percentile experiments completed."
