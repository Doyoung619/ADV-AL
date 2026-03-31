$ErrorActionPreference = "Stop"
$methods = @("badge", "badge_dual_a", "badge_dual_b")
$seeds = @(0,1,2)

$root = "./output_compare6"
New-Item -ItemType Directory -Force -Path $root | Out-Null

foreach ($method in $methods) {
  foreach ($seed in $seeds) {
    $runName = "compare6_${method}_seed${seed}"
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $runName"

    python main.py `
      --dataset cifar10 `
      --model small_cnn `
      --acquisition_method $method `
      --initial_labeled_size 200 `
      --acquisition_batch_size 50 `
      --rounds 20 `
      --epochs_per_round 50 `
      --epsilon_acq 0.0039215686 `
      --seed $seed `
      --output-dir $root `
      --run-name $runName

    if ($LASTEXITCODE -ne 0) {
      throw "Run failed: $runName (exit=$LASTEXITCODE)"
    }

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DONE  $runName"
  }
}

Write-Host "All output_compare6 runs completed."
