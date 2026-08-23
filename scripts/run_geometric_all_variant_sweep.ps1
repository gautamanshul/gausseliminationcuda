param(
    [string]$ExePath = "",
    [string[]]$Variants = @(
        "V1", "V2", "V3", "V3f", "V4", "V5a", "V5af", "V5bf",
        "VLU", "V6a", "V6b", "V6c", "V6d", "V6e", "V5c"
    ),
    [int[]]$Sizes = @(2000, 4000, 8000, 16000),
    [int]$TimedRepeats = 5,
    [int]$Block = 512,
    [int]$PanelWidth = 64,
    [int]$TileRows = 32,
    [int]$TileCols = 32,
    [int]$GpuIndex = 0,
    [int]$TelemetrySampleMs = 200,
    [string]$OutputPrefix = "",
    [string]$PlotPython = "",
    [switch]$SkipPlots
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputPrefix)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputPrefix = Join-Path $RepoRoot "results\geometric_all_variants_$Stamp"
} elseif (-not [System.IO.Path]::IsPathRooted($OutputPrefix)) {
    $OutputPrefix = Join-Path $RepoRoot $OutputPrefix
}

$ExpectedSizes = @(2000, 4000, 8000, 16000)
if (($Sizes -join ",") -ne ($ExpectedSizes -join ",")) {
    Write-Warning "The dissertation protocol uses n=1000*2^p for p=1..4: 2000,4000,8000,16000. This invocation overrides that ladder."
}

$SweepScript = Join-Path $PSScriptRoot "run_frozen_paired_sweep.ps1"
$Arguments = @{
    Variants = $Variants
    Sizes = $Sizes
    TimedRepeats = $TimedRepeats
    Block = $Block
    PanelWidth = $PanelWidth
    TileRows = $TileRows
    TileCols = $TileCols
    GpuIndex = $GpuIndex
    TelemetrySampleMs = $TelemetrySampleMs
    OutputPrefix = $OutputPrefix
}
if (-not [string]::IsNullOrWhiteSpace($ExePath)) {
    $Arguments.ExePath = $ExePath
}

Write-Host "Running geometric paired/interleaved scale sweep"
Write-Host "  n=$($Sizes -join ',')"
Write-Host "  variants=$($Variants -join ',')"
Write-Host "  output_prefix=$OutputPrefix"
& $SweepScript @Arguments

if (-not $SkipPlots) {
    $PlotScript = Join-Path $PSScriptRoot "plot_geometric_scale_sweep.py"
    if ([string]::IsNullOrWhiteSpace($PlotPython)) {
        $BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
        if (Test-Path -LiteralPath $BundledPython) {
            $PlotPython = $BundledPython
        } else {
            $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
            if ($null -ne $PythonCommand) {
                $PlotPython = $PythonCommand.Source
            }
        }
    }
    if ([string]::IsNullOrWhiteSpace($PlotPython)) {
        Write-Warning "Python was not found; the summary CSV is complete but plots were not generated."
    } else {
        & $PlotPython $PlotScript "${OutputPrefix}_summary.csv" --output-prefix $OutputPrefix
        if ($LASTEXITCODE -ne 0) {
            throw "Plot generation failed with exit code $LASTEXITCODE."
        }
    }
}
