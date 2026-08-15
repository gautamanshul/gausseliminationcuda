param(
    [string]$ExePath = "",
    [string[]]$Variants = @("V4", "V6c", "V6e", "V5c"),
    [int[]]$Sizes = @(2000, 4096, 6000, 8000),
    [int]$TimedRepeats = 5,
    [int]$Block = 512,
    [int]$PanelWidth = 64,
    [int]$TileRows = 32,
    [int]$TileCols = 32,
    [int]$GpuIndex = 0,
    [string]$OutputPrefix = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($ExePath)) {
    $ExePath = Join-Path $RepoRoot "out\build\x64-Release\gauss_elim_bench.exe"
}
$ExePath = (Resolve-Path -LiteralPath $ExePath).Path

if ($Variants.Count -eq 0) {
    throw "-Variants must contain at least one variant."
}
if ($Sizes.Count -eq 0 -or @($Sizes | Where-Object { $_ -le 0 }).Count -ne 0) {
    throw "-Sizes must contain positive matrix dimensions."
}
if ($TimedRepeats -le 0) {
    throw "-TimedRepeats must be positive."
}
if ($Block -le 0 -or $PanelWidth -le 0 -or $TileRows -le 0 -or $TileCols -le 0) {
    throw "Block, panel-width, and tile dimensions must be positive."
}

foreach ($Command in @("git", "cmake", "nvcc", "nvidia-smi")) {
    if ($null -eq (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "Required command is unavailable: $Command"
    }
}

$ResultsDir = Join-Path $RepoRoot "results"
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

if ([string]::IsNullOrWhiteSpace($OutputPrefix)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputPrefix = Join-Path $ResultsDir "paired_frozen_$Stamp"
} elseif (-not [System.IO.Path]::IsPathRooted($OutputPrefix)) {
    $OutputPrefix = Join-Path $RepoRoot $OutputPrefix
}

$OutputDirectory = Split-Path -Parent $OutputPrefix
if (-not [string]::IsNullOrWhiteSpace($OutputDirectory)) {
    New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
}

$TimedCsv = "${OutputPrefix}.csv"
$WarmupCsv = "${OutputPrefix}_warmup.csv"
$TelemetryCsv = "${OutputPrefix}_telemetry.csv"
$RunLog = "${OutputPrefix}_runlog.txt"
$SummaryCsv = "${OutputPrefix}_summary.csv"
$OutputFiles = @($TimedCsv, $WarmupCsv, $TelemetryCsv, $RunLog, $SummaryCsv)
$ExistingOutputs = @($OutputFiles | Where-Object { Test-Path -LiteralPath $_ })
if ($ExistingOutputs.Count -ne 0) {
    throw "Refusing to append to an existing sweep: $($ExistingOutputs -join ', ')"
}

function Get-VariantExtraArgs {
    param([Parameter(Mandatory)] [string]$Variant)

    if ($Variant -in @("V5a", "V5af", "V5b", "V5bf")) {
        return @("--tile-rows", [string]$TileRows, "--tile-cols", [string]$TileCols)
    }
    if ($Variant -in @("V6a", "V6b", "V6c", "V6d", "V6e", "V5c")) {
        return @("--panel-width", [string]$PanelWidth)
    }
    return @()
}

$Cases = @($Variants | ForEach-Object {
    [pscustomobject]@{
        Variant = $_
        ExtraArgs = @(Get-VariantExtraArgs -Variant $_)
    }
})

function Write-RunMetadata {
    "PAIRED_SWEEP start=$(Get-Date -Format o)" | Set-Content -LiteralPath $RunLog
    "repository=$RepoRoot" | Add-Content -LiteralPath $RunLog
    "executable=$ExePath" | Add-Content -LiteralPath $RunLog
    "variants=$($Variants -join ',')" | Add-Content -LiteralPath $RunLog
    "sizes=$($Sizes -join ',')" | Add-Content -LiteralPath $RunLog
    "timed_repeats=$TimedRepeats" | Add-Content -LiteralPath $RunLog
    "block=$Block" | Add-Content -LiteralPath $RunLog
    "panel_width=$PanelWidth" | Add-Content -LiteralPath $RunLog
    "tile=${TileRows}x${TileCols}" | Add-Content -LiteralPath $RunLog
    "gpu_index=$GpuIndex" | Add-Content -LiteralPath $RunLog
    "commit=$(& git -C $RepoRoot rev-parse HEAD)" | Add-Content -LiteralPath $RunLog
    "branch=$(& git -C $RepoRoot branch --show-current)" | Add-Content -LiteralPath $RunLog
    & nvidia-smi -i $GpuIndex | Add-Content -LiteralPath $RunLog
    & cmake --version | Add-Content -LiteralPath $RunLog
    & nvcc --version | Add-Content -LiteralPath $RunLog
}

function Invoke-PairedCase {
    param(
        [Parameter(Mandatory)] $Case,
        [Parameter(Mandatory)] [int]$N,
        [Parameter(Mandatory)] [string]$OutPath,
        [Parameter(Mandatory)] [string]$Stage,
        [int]$Repeat = -1
    )

    $Arguments = @(
        "--ablation",
        "--variant", $Case.Variant,
        "--n", [string]$N,
        "--block", [string]$Block,
        "--repeats", "1",
        "--cpu-reference-max-n", "0",
        "--out", $OutPath
    ) + $Case.ExtraArgs

    "CASE stage=$Stage repeat=$Repeat n=$N variant=$($Case.Variant) start=$(Get-Date -Format o)" |
        Tee-Object -FilePath $RunLog -Append
    & $ExePath @Arguments 2>&1 | Tee-Object -FilePath $RunLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark failed: stage=$Stage repeat=$Repeat n=$N variant=$($Case.Variant)"
    }
}

function Add-TelemetrySample {
    param(
        [Parameter(Mandatory)] $Case,
        [Parameter(Mandatory)] [int]$N,
        [Parameter(Mandatory)] [int]$Repeat,
        [Parameter(Mandatory)] [int]$LaunchPosition
    )

    $Line = & nvidia-smi -i $GpuIndex `
        --query-gpu=timestamp,name,driver_version,temperature.gpu,power.draw,clocks.gr,clocks.mem,utilization.gpu,utilization.memory `
        --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi telemetry failed for GPU index $GpuIndex."
    }
    $Sample = @($Line)[0] -split ",\s*"
    if ($Sample.Count -lt 9) {
        throw "Unexpected nvidia-smi telemetry row: $Line"
    }

    [pscustomobject]@{
        captured_at = Get-Date -Format o
        repeat = $Repeat
        n = $N
        launch_position = $LaunchPosition
        variant = $Case.Variant
        gpu_timestamp = $Sample[0]
        gpu_name = $Sample[1]
        driver_version = $Sample[2]
        temperature_c = $Sample[3]
        power_w = $Sample[4]
        graphics_clock_mhz = $Sample[5]
        memory_clock_mhz = $Sample[6]
        gpu_utilization_pct = $Sample[7]
        memory_utilization_pct = $Sample[8]
    } | Export-Csv -LiteralPath $TelemetryCsv -NoTypeInformation -Append
}

function Get-Percentile {
    param([double[]]$Values, [double]$P)

    $Sorted = @($Values | Sort-Object)
    if ($Sorted.Count -eq 1) {
        return $Sorted[0]
    }
    $Position = ($Sorted.Count - 1) * $P
    $Lower = [math]::Floor($Position)
    $Upper = [math]::Ceiling($Position)
    if ($Lower -eq $Upper) {
        return $Sorted[$Lower]
    }
    $Weight = $Position - $Lower
    return $Sorted[$Lower] * (1.0 - $Weight) + $Sorted[$Upper] * $Weight
}

function Test-AndSummarizeResults {
    $Rows = @(Import-Csv -LiteralPath $TimedCsv)
    $ExpectedRows = $Sizes.Count * $Cases.Count * $TimedRepeats
    if ($Rows.Count -ne $ExpectedRows) {
        throw "Expected $ExpectedRows timed rows; found $($Rows.Count)."
    }

    foreach ($Row in $Rows) {
        $GpuMs = [double]$Row.gpu_ms
        $Residual = [double]$Row.residual_norm2
        $SolutionError = [double]$Row.solution_error_norm2
        if ($GpuMs -le 0 -or
            [double]::IsNaN($GpuMs) -or [double]::IsInfinity($GpuMs) -or
            [double]::IsNaN($Residual) -or [double]::IsInfinity($Residual) -or
            [double]::IsNaN($SolutionError) -or [double]::IsInfinity($SolutionError)) {
            throw "Invalid timed row: variant=$($Row.variant) n=$($Row.n)"
        }
    }

    $Summary = $Rows |
        Group-Object variant,n |
        ForEach-Object {
            $GroupRows = @($_.Group)
            [double[]]$Times = $GroupRows | ForEach-Object { [double]$_.gpu_ms }
            [double[]]$Residuals = $GroupRows | ForEach-Object { [double]$_.residual_norm2 }
            [double[]]$SolutionErrors = $GroupRows | ForEach-Object { [double]$_.solution_error_norm2 }
            $Q1 = Get-Percentile -Values $Times -P 0.25
            $Median = Get-Percentile -Values $Times -P 0.50
            $Q3 = Get-Percentile -Values $Times -P 0.75

            [pscustomobject]@{
                variant = $GroupRows[0].variant
                n = [int]$GroupRows[0].n
                repeats = $GroupRows.Count
                median_gpu_ms = [math]::Round($Median, 6)
                q1_gpu_ms = [math]::Round($Q1, 6)
                q3_gpu_ms = [math]::Round($Q3, 6)
                iqr_gpu_ms = [math]::Round($Q3 - $Q1, 6)
                min_gpu_ms = [math]::Round(($Times | Measure-Object -Minimum).Minimum, 6)
                max_gpu_ms = [math]::Round(($Times | Measure-Object -Maximum).Maximum, 6)
                max_residual_norm2 = ($Residuals | Measure-Object -Maximum).Maximum
                max_solution_error_norm2 = ($SolutionErrors | Measure-Object -Maximum).Maximum
            }
        } |
        Sort-Object n, variant

    $Summary | Export-Csv -LiteralPath $SummaryCsv -NoTypeInformation
    $Summary | Format-Table -AutoSize
}

Push-Location $RepoRoot
try {
    Write-RunMetadata

    # Warm-ups are measured by the executable but excluded from the timed dataset.
    foreach ($N in $Sizes) {
        foreach ($Case in $Cases) {
            Invoke-PairedCase -Case $Case -N $N -OutPath $WarmupCsv -Stage "warmup"
        }
    }

    # Rotate the first variant on every repeat to reduce launch-position bias.
    for ($Repeat = 0; $Repeat -lt $TimedRepeats; $Repeat++) {
        foreach ($N in $Sizes) {
            $Offset = $Repeat % $Cases.Count
            $OrderedCases = @(for ($Index = 0; $Index -lt $Cases.Count; $Index++) {
                $Cases[($Index + $Offset) % $Cases.Count]
            })

            for ($LaunchPosition = 0; $LaunchPosition -lt $OrderedCases.Count; $LaunchPosition++) {
                $Case = $OrderedCases[$LaunchPosition]
                Add-TelemetrySample -Case $Case -N $N -Repeat $Repeat `
                    -LaunchPosition $LaunchPosition
                Invoke-PairedCase -Case $Case -N $N -OutPath $TimedCsv `
                    -Stage "timed" -Repeat $Repeat
            }
        }
    }

    Test-AndSummarizeResults
    "PAIRED_SWEEP done=$(Get-Date -Format o)" | Add-Content -LiteralPath $RunLog
    Write-Host "Timed rows:  $TimedCsv"
    Write-Host "Warm-up rows: $WarmupCsv"
    Write-Host "Telemetry:    $TelemetryCsv"
    Write-Host "Summary:      $SummaryCsv"
    Write-Host "Run log:      $RunLog"
} catch {
    "PAIRED_SWEEP failed=$(Get-Date -Format o) error=$($_.Exception.Message)" |
        Add-Content -LiteralPath $RunLog
    throw
} finally {
    Pop-Location
}
