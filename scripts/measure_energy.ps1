param(
    [string]$ExePath = ".\out\build\x64-Release\gauss_elim_bench.exe",
    [string]$OutPath = "results\energy_pilot.csv",
    [string]$BenchmarkOutPath = "results\energy_pilot_benchmark_rows.csv",
    [string]$WarmupOutPath = "results\energy_pilot_warmup.csv",
    [string[]]$Variants = @("V4", "V6a", "V6b", "V3f", "V5af"),
    [int[]]$Sizes = @(4000, 6000),
    [int]$Repeats = 3,
    [int]$Block = 512,
    [int]$PanelWidth = 64,
    [int]$TileRows = 32,
    [int]$TileCols = 32,
    [int]$SampleMs = 50,
    [int]$WarmupN = 512,
    [int]$CaseRepeats = 1
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($CaseRepeats -le 0) {
    throw "-CaseRepeats must be positive."
}

function Get-GpuSample {
    $line = & nvidia-smi --query-gpu=power.draw,temperature.gpu,clocks.gr,clocks.mem,utilization.gpu,utilization.memory --format=csv,noheader,nounits
    $parts = $line -split "," | ForEach-Object { $_.Trim() }
    if ($parts.Count -lt 6) {
        throw "Unexpected nvidia-smi sample: $line"
    }
    [pscustomobject]@{
        PowerW = [double]$parts[0]
        TempC = [double]$parts[1]
        GraphicsClockMHz = [double]$parts[2]
        MemoryClockMHz = [double]$parts[3]
        GpuUtilPct = [double]$parts[4]
        MemUtilPct = [double]$parts[5]
    }
}

function Add-VariantArgs {
    param([string[]]$BaseArgs, [string]$Variant)
    if ($Variant -eq "V5af") {
        return $BaseArgs + @("--tile-rows", [string]$TileRows, "--tile-cols", [string]$TileCols)
    }
    if ($Variant -eq "V6a" -or $Variant -eq "V6b" -or $Variant -eq "V6c") {
        return $BaseArgs + @("--panel-width", [string]$PanelWidth)
    }
    return $BaseArgs
}

function Run-BenchmarkCase {
    param(
        [string]$Variant,
        [int]$N,
        [int]$RunIndex
    )

    $warmArgs = @(
        "--ablation", "--variant", $Variant,
        "--n", [string]$WarmupN,
        "--block", [string]$Block,
        "--cpu-reference-max-n", "0",
        "--repeats", "1",
        "--out", $WarmupOutPath
    )
    $warmArgs = Add-VariantArgs -BaseArgs $warmArgs -Variant $Variant
    & $ExePath @warmArgs | Out-Host

    $caseArgs = @(
        "--ablation", "--variant", $Variant,
        "--n", [string]$N,
        "--block", [string]$Block,
        "--cpu-reference-max-n", "0",
        "--repeats", [string]$CaseRepeats,
        "--out", $BenchmarkOutPath
    )
    $caseArgs = Add-VariantArgs -BaseArgs $caseArgs -Variant $Variant

    $stdout = [System.IO.Path]::GetTempFileName()
    $stderr = [System.IO.Path]::GetTempFileName()
    $samples = New-Object System.Collections.Generic.List[object]
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    $process = Start-Process -FilePath $ExePath -ArgumentList $caseArgs -NoNewWindow -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    while (-not $process.HasExited) {
        $sample = Get-GpuSample
        $samples.Add([pscustomobject]@{
            T = $stopwatch.Elapsed.TotalSeconds
            PowerW = $sample.PowerW
            TempC = $sample.TempC
            GraphicsClockMHz = $sample.GraphicsClockMHz
            MemoryClockMHz = $sample.MemoryClockMHz
            GpuUtilPct = $sample.GpuUtilPct
            MemUtilPct = $sample.MemUtilPct
        })
        Start-Sleep -Milliseconds $SampleMs
    }
    $process.WaitForExit()
    $process.Refresh()
    $durationS = $stopwatch.Elapsed.TotalSeconds

    Get-Content -LiteralPath $stdout | Out-Host
    $errText = Get-Content -LiteralPath $stderr -ErrorAction SilentlyContinue
    if ($errText) {
        $errText | Out-Host
    }
    Remove-Item -LiteralPath $stdout, $stderr -Force -ErrorAction SilentlyContinue

    if ($null -ne $process.ExitCode -and $process.ExitCode -ne 0) {
        throw "Benchmark failed for variant=$Variant n=$N run=$RunIndex exit=$($process.ExitCode)"
    }

    if ($samples.Count -eq 0) {
        $sample = Get-GpuSample
        $samples.Add([pscustomobject]@{
            T = 0.0
            PowerW = $sample.PowerW
            TempC = $sample.TempC
            GraphicsClockMHz = $sample.GraphicsClockMHz
            MemoryClockMHz = $sample.MemoryClockMHz
            GpuUtilPct = $sample.GpuUtilPct
            MemUtilPct = $sample.MemUtilPct
        })
    }

    $energyJ = 0.0
    for ($i = 0; $i -lt $samples.Count; $i++) {
        if ($i -lt $samples.Count - 1) {
            $dt = [double]$samples[$i + 1].T - [double]$samples[$i].T
        } else {
            $dt = [Math]::Max(0.0, $durationS - [double]$samples[$i].T)
        }
        $energyJ += [double]$samples[$i].PowerW * $dt
    }

    $avgPowerW = if ($durationS -gt 0.0) { $energyJ / $durationS } else { 0.0 }
    $maxPowerW = ($samples | Measure-Object -Property PowerW -Maximum).Maximum
    $avgTempC = ($samples | Measure-Object -Property TempC -Average).Average
    $maxTempC = ($samples | Measure-Object -Property TempC -Maximum).Maximum
    $avgGpuUtil = ($samples | Measure-Object -Property GpuUtilPct -Average).Average
    $maxGpuUtil = ($samples | Measure-Object -Property GpuUtilPct -Maximum).Maximum
    $avgMemUtil = ($samples | Measure-Object -Property MemUtilPct -Average).Average
    $avgGraphicsClock = ($samples | Measure-Object -Property GraphicsClockMHz -Average).Average
    $avgMemoryClock = ($samples | Measure-Object -Property MemoryClockMHz -Average).Average

    $benchRows = @(Import-Csv -LiteralPath $BenchmarkOutPath)
    $caseRows = @($benchRows | Select-Object -Last $CaseRepeats)
    $bench = $caseRows[-1]
    $gpuMsValues = @($caseRows | ForEach-Object { [double]$_.gpu_ms } | Sort-Object)
    $gpuMid = [int]($gpuMsValues.Count / 2)
    $medianGpuMs = if ($gpuMsValues.Count % 2) {
        $gpuMsValues[$gpuMid]
    } else {
        ($gpuMsValues[$gpuMid - 1] + $gpuMsValues[$gpuMid]) / 2.0
    }
    $totalEffectiveGflop = 0.0
    foreach ($caseRow in $caseRows) {
        $totalEffectiveGflop += [double]$caseRow.effective_gflops *
            ([double]$caseRow.gpu_ms / 1000.0)
    }
    $medianEffectiveGflopsValues = @($caseRows | ForEach-Object { [double]$_.effective_gflops } | Sort-Object)
    $gflopsMid = [int]($medianEffectiveGflopsValues.Count / 2)
    $medianEffectiveGflops = if ($medianEffectiveGflopsValues.Count % 2) {
        $medianEffectiveGflopsValues[$gflopsMid]
    } else {
        ($medianEffectiveGflopsValues[$gflopsMid - 1] +
            $medianEffectiveGflopsValues[$gflopsMid]) / 2.0
    }
    $joulesPerEffectiveGflop = if ($totalEffectiveGflop -gt 0.0) { $energyJ / $totalEffectiveGflop } else { "" }

    [pscustomobject]@{
        timestamp = (Get-Date).ToString("s")
        variant = $Variant
        requested_variant = $Variant
        csv_variant = $bench.variant
        n = [int]$bench.n
        run_index = $RunIndex
        case_repeats = $CaseRepeats
        sample_ms = $SampleMs
        sample_count = $samples.Count
        process_wall_ms = [Math]::Round($durationS * 1000.0, 3)
        median_gpu_ms = [Math]::Round($medianGpuMs, 6)
        total_gpu_ms = [Math]::Round((($caseRows | ForEach-Object { [double]$_.gpu_ms }) | Measure-Object -Sum).Sum, 6)
        median_effective_gflops = [Math]::Round($medianEffectiveGflops, 6)
        total_effective_gflop = [Math]::Round($totalEffectiveGflop, 6)
        energy_j = [Math]::Round($energyJ, 6)
        avg_power_w = [Math]::Round($avgPowerW, 6)
        max_power_w = [Math]::Round([double]$maxPowerW, 6)
        joules_per_effective_gflop = if ($joulesPerEffectiveGflop -eq "") { "" } else { [Math]::Round($joulesPerEffectiveGflop, 6) }
        avg_temp_c = [Math]::Round([double]$avgTempC, 3)
        max_temp_c = [Math]::Round([double]$maxTempC, 3)
        avg_gpu_util_pct = [Math]::Round([double]$avgGpuUtil, 3)
        max_gpu_util_pct = [Math]::Round([double]$maxGpuUtil, 3)
        avg_mem_util_pct = [Math]::Round([double]$avgMemUtil, 3)
        avg_graphics_clock_mhz = [Math]::Round([double]$avgGraphicsClock, 3)
        avg_memory_clock_mhz = [Math]::Round([double]$avgMemoryClock, 3)
        residual_norm2 = [double]$bench.residual_norm2
        solution_error_norm2 = [double]$bench.solution_error_norm2
    }
}

$outDir = Split-Path -Parent $OutPath
if ($outDir -and -not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

$allRows = New-Object System.Collections.Generic.List[object]
for ($run = 0; $run -lt $Repeats; $run++) {
    foreach ($n in $Sizes) {
        for ($j = 0; $j -lt $Variants.Count; $j++) {
            $variant = $Variants[($j + $run) % $Variants.Count]
            Write-Host "ENERGY_CASE variant=$variant n=$n run=$run"
            $row = Run-BenchmarkCase -Variant $variant -N $n -RunIndex $run
            $allRows.Add($row)
            $allRows | Export-Csv -LiteralPath $OutPath -NoTypeInformation
        }
    }
}

Write-Host "Wrote energy results to $OutPath"
