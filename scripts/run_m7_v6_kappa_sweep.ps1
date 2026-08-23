param(
    [string]$ExePath = "",
    [string[]]$Variants = @("V4", "V6c", "V6e", "V5c"),
    [int[]]$Sizes = @(2000, 4096, 6000, 8000),
    [double[]]$Kappas = @(1.0e2, 1.0e4, 1.0e6),
    [int]$TotalRepeats = 7,
    [int]$DiscardRepeats = 1,
    [int]$Block = 512,
    [int]$PanelWidth = 64,
    [int]$TileRows = 32,
    [int]$TileCols = 32,
    [int]$BaseSeed = 42,
    [int]$CpuReferenceMaxN = 0,
    [int]$GpuIndex = 0,
    [int]$TelemetrySampleMs = 200,
    [string]$OutputPrefix = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($ExePath)) {
    $ExePath = Join-Path $RepoRoot "out\build\x64-Release\gauss_elim_bench.exe"
}
$ExePath = (Resolve-Path -LiteralPath $ExePath).Path

if ($Variants.Count -eq 0 -or $Sizes.Count -eq 0 -or $Kappas.Count -eq 0) {
    throw "Variants, Sizes, and Kappas must not be empty."
}
if (@($Sizes | Where-Object { $_ -le 0 }).Count -ne 0) {
    throw "Every matrix size must be positive."
}
if (@($Kappas | Where-Object { $_ -lt 1.0 }).Count -ne 0) {
    throw "Every kappa must be at least 1."
}
if ($TotalRepeats -le 0 -or $DiscardRepeats -lt 0 -or
    $DiscardRepeats -ge $TotalRepeats) {
    throw "Require TotalRepeats > 0 and 0 <= DiscardRepeats < TotalRepeats."
}
if ($Block -le 0 -or $PanelWidth -le 0 -or $TileRows -le 0 -or
    $TileCols -le 0 -or $TelemetrySampleMs -le 0) {
    throw "Block, PanelWidth, tile dimensions, and TelemetrySampleMs must be positive."
}
if ($CpuReferenceMaxN -lt 0) {
    throw "CpuReferenceMaxN must be non-negative."
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
    $OutputPrefix = Join-Path $ResultsDir "m7_v6family_kappa_sweep_$Stamp"
} elseif (-not [System.IO.Path]::IsPathRooted($OutputPrefix)) {
    $OutputPrefix = Join-Path $RepoRoot $OutputPrefix
}

$OutputDirectory = Split-Path -Parent $OutputPrefix
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$RawCsv = "${OutputPrefix}.csv"
$ManifestCsv = "${OutputPrefix}_protocol_manifest.csv"
$TelemetryCsv = "${OutputPrefix}_telemetry.csv"
$SummaryCsv = "${OutputPrefix}_summary.csv"
$SummaryMd = "${OutputPrefix}_summary.md"
$RunLog = "${OutputPrefix}_runlog.txt"
$OutputFiles = @($RawCsv, $ManifestCsv, $TelemetryCsv, $SummaryCsv,
                 $SummaryMd, $RunLog)
$Existing = @($OutputFiles | Where-Object { Test-Path -LiteralPath $_ })
if ($Existing.Count -ne 0) {
    throw "Refusing to overwrite or append to existing output: $($Existing -join ', ')"
}

function ConvertTo-InvariantNumber {
    param([Parameter(Mandatory)] [string]$Value)

    $Number = 0.0
    if (-not [double]::TryParse(
            $Value,
            [System.Globalization.NumberStyles]::Float,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [ref]$Number)) {
        throw "Not a valid floating-point value: $Value"
    }
    return $Number
}

function Test-Finite {
    param([Parameter(Mandatory)] [double]$Value)
    return -not ([double]::IsNaN($Value) -or [double]::IsInfinity($Value))
}

function Get-Percentile {
    param([Parameter(Mandatory)] [double[]]$Values,
          [Parameter(Mandatory)] [double]$P)

    $Sorted = @($Values | Sort-Object)
    if ($Sorted.Count -eq 1) { return $Sorted[0] }
    $Position = ($Sorted.Count - 1) * $P
    $Lower = [math]::Floor($Position)
    $Upper = [math]::Ceiling($Position)
    if ($Lower -eq $Upper) { return $Sorted[$Lower] }
    $Weight = $Position - $Lower
    return $Sorted[$Lower] * (1.0 - $Weight) + $Sorted[$Upper] * $Weight
}

function Add-TelemetrySample {
    param(
        [Parameter(Mandatory)] [int]$Sequence,
        [Parameter(Mandatory)] [int]$Repeat,
        [Parameter(Mandatory)] [int]$N,
        [Parameter(Mandatory)] [double]$Kappa,
        [Parameter(Mandatory)] [string]$Variant,
        [Parameter(Mandatory)] [int]$LaunchPosition
    )

    $Line = & nvidia-smi -i $GpuIndex `
        --query-gpu=timestamp,name,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory `
        --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi telemetry query failed."
    }
    $Parts = @($Line)[0] -split ",\s*"
    if ($Parts.Count -lt 9) {
        throw "Unexpected nvidia-smi telemetry row: $Line"
    }
    [pscustomobject]@{
        captured_at = Get-Date -Format o
        sequence = $Sequence
        protocol_repeat = $Repeat
        stage = if ($Repeat -lt $DiscardRepeats) { "warmup" } else { "record" }
        n = $N
        kappa = $Kappa.ToString("G17", [System.Globalization.CultureInfo]::InvariantCulture)
        requested_variant = $Variant
        launch_position = $LaunchPosition
        gpu_timestamp = $Parts[0]
        gpu_name = $Parts[1]
        driver_version = $Parts[2]
        temperature_c = $Parts[3]
        power_w = $Parts[4]
        sm_clock_mhz = $Parts[5]
        memory_clock_mhz = $Parts[6]
        gpu_utilization_pct = $Parts[7]
        memory_utilization_pct = $Parts[8]
    } | Export-Csv -LiteralPath $TelemetryCsv -NoTypeInformation -Append
}

function Get-RawRowCount {
    if (-not (Test-Path -LiteralPath $RawCsv)) { return 0 }
    return @(Import-Csv -LiteralPath $RawCsv).Count
}

function Invoke-M7Case {
    param(
        [Parameter(Mandatory)] [int]$Sequence,
        [Parameter(Mandatory)] [int]$Repeat,
        [Parameter(Mandatory)] [int]$N,
        [Parameter(Mandatory)] [double]$Kappa,
        [Parameter(Mandatory)] [string]$Variant,
        [Parameter(Mandatory)] [int]$LaunchPosition
    )

    $Stage = if ($Repeat -lt $DiscardRepeats) { "warmup" } else { "record" }
    $KappaArg = $Kappa.ToString("G17", [System.Globalization.CultureInfo]::InvariantCulture)
    # Each process receives the paired repeat's seed. The executable adds its
    # deterministic n/kappa term, so every variant in a cell gets one matrix.
    $ProtocolSeed = $BaseSeed + 1000003 * $Repeat
    $Arguments = @(
        "--ablation", "--m7-synthetic",
        "--variant", $Variant,
        "--n", [string]$N,
        "--kappa", $KappaArg,
        "--repeats", "1",
        "--seed", [string]$ProtocolSeed,
        "--block", [string]$Block,
        "--panel-width", [string]$PanelWidth,
        "--tile-rows", [string]$TileRows,
        "--tile-cols", [string]$TileCols,
        "--m7-cpu-reference-max-n", [string]$CpuReferenceMaxN,
        "--out", $RawCsv
    )
    $Stdout = [System.IO.Path]::GetTempFileName()
    $Stderr = [System.IO.Path]::GetTempFileName()
    $BeforeRows = Get-RawRowCount
    $StartedAt = Get-Date
    $Watch = [System.Diagnostics.Stopwatch]::StartNew()
    $Status = "ok"
    $Failure = ""

    "CASE sequence=$Sequence stage=$Stage repeat=$Repeat n=$N kappa=$KappaArg variant=$Variant launch_position=$LaunchPosition start=$($StartedAt.ToString('o'))" |
        Tee-Object -FilePath $RunLog -Append
    try {
        $Process = Start-Process -FilePath $ExePath -ArgumentList $Arguments `
            -NoNewWindow -PassThru -RedirectStandardOutput $Stdout `
            -RedirectStandardError $Stderr
        while (-not $Process.HasExited) {
            Add-TelemetrySample -Sequence $Sequence -Repeat $Repeat -N $N `
                -Kappa $Kappa -Variant $Variant -LaunchPosition $LaunchPosition
            Start-Sleep -Milliseconds $TelemetrySampleMs
            $Process.Refresh()
        }
        $Process.WaitForExit()
        $Process.Refresh()
        if ($null -ne $Process.ExitCode -and $Process.ExitCode -ne 0) {
            $Status = "process_failed"
            $Failure = "exit_code=$($Process.ExitCode)"
        }
    } catch {
        $Status = "driver_failed"
        $Failure = $_.Exception.Message
    } finally {
        $Watch.Stop()
        Get-Content -LiteralPath $Stdout -ErrorAction SilentlyContinue |
            Tee-Object -FilePath $RunLog -Append | Write-Host
        Get-Content -LiteralPath $Stderr -ErrorAction SilentlyContinue |
            Tee-Object -FilePath $RunLog -Append | Write-Host
        Remove-Item -LiteralPath $Stdout, $Stderr -Force -ErrorAction SilentlyContinue
    }

    $AfterRows = Get-RawRowCount
    $RawRowIndex = ""
    if ($AfterRows -eq $BeforeRows + 1) {
        $RawRowIndex = $AfterRows - 1
        $Row = @(Import-Csv -LiteralPath $RawCsv)[-1]
        $GpuMs = ConvertTo-InvariantNumber $Row.gpu_ms
        $Residual = ConvertTo-InvariantNumber $Row.residual_norm2
        $SolutionError = ConvertTo-InvariantNumber $Row.solution_error_norm2
        if ($GpuMs -le 0 -or -not (Test-Finite $GpuMs) -or
            -not (Test-Finite $Residual) -or -not (Test-Finite $SolutionError)) {
            $Status = "correctness_failed"
            $Failure = "non-positive time or non-finite correctness metric"
        } elseif ($Kappa -le 1.0e4 -and $Residual -ge 1.0e-4) {
            $Status = "correctness_failed"
            $Failure = "residual_norm2=$Residual exceeded 1e-4"
        }
    } elseif ($Status -eq "ok") {
        $Status = "row_count_failed"
        $Failure = "expected one new row; before=$BeforeRows after=$AfterRows"
    }

    [pscustomobject]@{
        sequence = $Sequence
        protocol_repeat = $Repeat
        stage = $Stage
        n = $N
        kappa = $KappaArg
        requested_variant = $Variant
        launch_position = $LaunchPosition
        protocol_seed = $ProtocolSeed
        raw_row_index = $RawRowIndex
        started_at = $StartedAt.ToString("o")
        process_wall_ms = [math]::Round($Watch.Elapsed.TotalMilliseconds, 3)
        status = $Status
        failure = $Failure
    } | Export-Csv -LiteralPath $ManifestCsv -NoTypeInformation -Append
}

function Write-RunMetadata {
    "M7_V6_KAPPA_SWEEP start=$(Get-Date -Format o)" | Set-Content -LiteralPath $RunLog
    "repository=$RepoRoot" | Add-Content -LiteralPath $RunLog
    "executable=$ExePath" | Add-Content -LiteralPath $RunLog
    "commit=$(& git -C $RepoRoot rev-parse HEAD)" | Add-Content -LiteralPath $RunLog
    "branch=$(& git -C $RepoRoot branch --show-current)" | Add-Content -LiteralPath $RunLog
    "variants=$($Variants -join ',')" | Add-Content -LiteralPath $RunLog
    "sizes=$($Sizes -join ',')" | Add-Content -LiteralPath $RunLog
    "kappas=$($Kappas -join ',')" | Add-Content -LiteralPath $RunLog
    "total_repeats=$TotalRepeats" | Add-Content -LiteralPath $RunLog
    "discard_repeats=$DiscardRepeats" | Add-Content -LiteralPath $RunLog
    "base_seed=$BaseSeed" | Add-Content -LiteralPath $RunLog
    "tile=${TileRows}x${TileCols}" | Add-Content -LiteralPath $RunLog
    "cpu_reference_max_n=$CpuReferenceMaxN" | Add-Content -LiteralPath $RunLog
    & nvidia-smi -i $GpuIndex | Add-Content -LiteralPath $RunLog
    & cmake --version | Add-Content -LiteralPath $RunLog
    & nvcc --version | Add-Content -LiteralPath $RunLog
}

function Write-Summary {
    $RawRows = @(Import-Csv -LiteralPath $RawCsv)
    $Manifest = @(Import-Csv -LiteralPath $ManifestCsv)
    $Expected = $TotalRepeats * $Sizes.Count * $Kappas.Count * $Variants.Count
    if ($Manifest.Count -ne $Expected) {
        throw "Expected $Expected manifest rows; found $($Manifest.Count)."
    }

    $Joined = foreach ($Entry in $Manifest) {
        if ($Entry.status -ne "ok" -or [string]::IsNullOrWhiteSpace($Entry.raw_row_index)) {
            continue
        }
        $Row = $RawRows[[int]$Entry.raw_row_index]
        [pscustomobject]@{
            protocol_repeat = [int]$Entry.protocol_repeat
            stage = $Entry.stage
            n = [int]$Entry.n
            kappa = ConvertTo-InvariantNumber $Entry.kappa
            requested_variant = $Entry.requested_variant
            observed_variant = $Row.variant
            gpu_ms = ConvertTo-InvariantNumber $Row.gpu_ms
            residual_norm2 = ConvertTo-InvariantNumber $Row.residual_norm2
            solution_error_norm2 = ConvertTo-InvariantNumber $Row.solution_error_norm2
        }
    }
    $RecordRows = @($Joined | Where-Object { $_.stage -eq "record" })
    $Summary = $RecordRows |
        Group-Object requested_variant,n,kappa |
        ForEach-Object {
            $Rows = @($_.Group)
            [double[]]$Times = $Rows | ForEach-Object { $_.gpu_ms }
            [double[]]$Residuals = $Rows | ForEach-Object { $_.residual_norm2 }
            [double[]]$Errors = $Rows | ForEach-Object { $_.solution_error_norm2 }
            $Q1 = Get-Percentile -Values $Times -P 0.25
            $Median = Get-Percentile -Values $Times -P 0.50
            $Q3 = Get-Percentile -Values $Times -P 0.75
            [pscustomobject]@{
                variant = $Rows[0].requested_variant
                observed_variant = $Rows[0].observed_variant
                n = $Rows[0].n
                kappa = $Rows[0].kappa
                repeats_of_record = $Rows.Count
                median_gpu_ms = [math]::Round($Median, 6)
                q1_gpu_ms = [math]::Round($Q1, 6)
                q3_gpu_ms = [math]::Round($Q3, 6)
                iqr_gpu_ms = [math]::Round($Q3 - $Q1, 6)
                min_residual_norm2 = ($Residuals | Measure-Object -Minimum).Minimum
                max_residual_norm2 = ($Residuals | Measure-Object -Maximum).Maximum
                min_solution_error_norm2 = ($Errors | Measure-Object -Minimum).Minimum
                max_solution_error_norm2 = ($Errors | Measure-Object -Maximum).Maximum
            }
        } | Sort-Object n, kappa, variant
    $Summary | Export-Csv -LiteralPath $SummaryCsv -NoTypeInformation

    $Commit = & git -C $RepoRoot rev-parse HEAD
    $Gpu = (& nvidia-smi -i $GpuIndex --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader,nounits) -join ""
    $Cuda = (& nvcc --version | Select-Object -Last 1) -join ""
    $Failed = @($Manifest | Where-Object { $_.status -ne "ok" })
    $Lines = New-Object System.Collections.Generic.List[string]
    $Lines.Add("# M7 conditioned-system kappa sweep summary")
    $Lines.Add("")
    $Lines.Add("**Generated:** $(Get-Date -Format o)")
    $Lines.Add("**Commit:** ``$Commit``")
    $Lines.Add("**GPU:** $Gpu")
    $Lines.Add("**CUDA:** $Cuda")
    $Lines.Add("**Protocol:** $TotalRepeats paired/interleaved repeats; first $DiscardRepeats discarded; panel width $PanelWidth; tile ${TileRows}x${TileCols}; CPU reference gate $CpuReferenceMaxN (known M7 solution used for correctness).")
    $Lines.Add("")
    $Lines.Add("The raw benchmark CSV is preserved exactly as emitted by the executable. The companion manifest records outer repeat, warm-up/record status, rotating launch position, and raw-row index.")
    $Lines.Add("")
    $Lines.Add("## Median timing and correctness ranges")
    $Lines.Add("")
    $Lines.Add("| Variant | n | kappa | Repeats | Median GPU ms | IQR ms | Residual norm2 range | Solution-error norm2 range |")
    $Lines.Add("|---|---:|---:|---:|---:|---:|---:|---:|")
    foreach ($Row in $Summary) {
        $Lines.Add("| $($Row.variant) | $($Row.n) | $($Row.kappa.ToString('G6')) | $($Row.repeats_of_record) | $($Row.median_gpu_ms) | $($Row.iqr_gpu_ms) | $($Row.min_residual_norm2)-$($Row.max_residual_norm2) | $($Row.min_solution_error_norm2)-$($Row.max_solution_error_norm2) |")
    }
    $Lines.Add("")
    $Lines.Add("## Correctness gate")
    $Lines.Add("")
    if ($Failed.Count -eq 0) {
        $Lines.Add("All $($Manifest.Count) attempted solves produced positive timings and finite residual/solution-error metrics; well/moderately conditioned residuals passed the 1e-4 gate.")
    } else {
        $Lines.Add("$($Failed.Count) attempted cases were flagged. They remain in the protocol manifest and are excluded from timing summaries:")
        foreach ($Failure in $Failed) {
            $Lines.Add("- repeat=$($Failure.protocol_repeat), n=$($Failure.n), kappa=$($Failure.kappa), variant=$($Failure.requested_variant): $($Failure.status), $($Failure.failure)")
        }
    }
    $Lines.Add("")
    $Lines.Add("## FP32 conditioning interpretation")
    $Lines.Add("")
    $Lines.Add("Interpret solution-error growth against the rough sensitivity scale kappa times FP32 machine epsilon (about 1.19e-7). A finite increase at kappa=1e6 is expected; NaN/Inf, solver failure, or residual-gate failure is reported as a breakdown rather than removed.")
    $Lines.Add("")
    $Lines.Add("## V6c runtime decomposition")
    $Lines.Add("")
    $Lines.Add("Pending Nsight Systems gpukernsum and timeline analysis. Add measured kernel totals here without changing the sweep rows.")
    $Lines | Set-Content -LiteralPath $SummaryMd -Encoding ASCII

    $Summary | Format-Table -AutoSize
    if ($Failed.Count -ne 0) {
        throw "$($Failed.Count) cases failed the process or correctness gate; inspect $ManifestCsv."
    }
}

Push-Location $RepoRoot
try {
    Write-RunMetadata
    $Sequence = 0
    for ($Repeat = 0; $Repeat -lt $TotalRepeats; $Repeat++) {
        foreach ($N in $Sizes) {
            foreach ($Kappa in $Kappas) {
                # Rotate the first variant each repeat to distribute thermal and
                # launch-position effects without breaking within-cell pairing.
                $Offset = $Repeat % $Variants.Count
                $OrderedVariants = @(for ($Index = 0; $Index -lt $Variants.Count; $Index++) {
                    $Variants[($Index + $Offset) % $Variants.Count]
                })
                for ($LaunchPosition = 0; $LaunchPosition -lt $OrderedVariants.Count; $LaunchPosition++) {
                    Invoke-M7Case -Sequence $Sequence -Repeat $Repeat -N $N `
                        -Kappa $Kappa -Variant $OrderedVariants[$LaunchPosition] `
                        -LaunchPosition $LaunchPosition
                    $Sequence++
                }
            }
        }
    }
    Write-Summary
    "M7_V6_KAPPA_SWEEP done=$(Get-Date -Format o)" | Add-Content -LiteralPath $RunLog
    Write-Host "Raw CSV:    $RawCsv"
    Write-Host "Manifest:   $ManifestCsv"
    Write-Host "Telemetry:  $TelemetryCsv"
    Write-Host "Summary:    $SummaryCsv"
    Write-Host "Summary MD: $SummaryMd"
    Write-Host "Run log:    $RunLog"
} catch {
    "M7_V6_KAPPA_SWEEP failed=$(Get-Date -Format o) error=$($_.Exception.Message)" |
        Add-Content -LiteralPath $RunLog
    throw
} finally {
    Pop-Location
}
