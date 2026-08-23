param(
    [string]$ExePath = "",
    [string[]]$Variants = @(
        "V1", "V2", "V3", "V3f", "V4", "V5a", "V5af", "V5bf",
        "VLU", "V6a", "V6b", "V6c", "V6d", "V6e", "V5c"
    ),
    [int[]]$Sizes = @(2000, 4000),
    [double[]]$Kappas = @(1.0e2, 1.0e4, 1.0e6),
    [int]$TotalRepeats = 7,
    [int]$DiscardRepeats = 1,
    [int]$Block = 512,
    [int]$PanelWidth = 64,
    [int]$TileRows = 32,
    [int]$TileCols = 32,
    [int]$BaseSeed = 42,
    [int]$GpuIndex = 0,
    [string]$OutputPrefix = "",
    [string]$PlotPython = "",
    [switch]$SkipPlots
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputPrefix)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputPrefix = Join-Path $RepoRoot "results\geometric_conditioning_$Stamp"
} elseif (-not [System.IO.Path]::IsPathRooted($OutputPrefix)) {
    $OutputPrefix = Join-Path $RepoRoot $OutputPrefix
}

if (@($Sizes | Where-Object { $_ -gt 4000 }).Count -ne 0) {
    Write-Warning "The conditioned SPD generator is O(n^2) in memory and performs repeated dense rotations. Sizes above 4000 are exploratory, not part of the default bounded conditioning protocol."
}

$SweepScript = Join-Path $PSScriptRoot "run_m7_v6_kappa_sweep.ps1"
$Arguments = @{
    Variants = $Variants
    Sizes = $Sizes
    Kappas = $Kappas
    TotalRepeats = $TotalRepeats
    DiscardRepeats = $DiscardRepeats
    Block = $Block
    PanelWidth = $PanelWidth
    TileRows = $TileRows
    TileCols = $TileCols
    BaseSeed = $BaseSeed
    CpuReferenceMaxN = 0
    GpuIndex = $GpuIndex
    OutputPrefix = $OutputPrefix
}
if (-not [string]::IsNullOrWhiteSpace($ExePath)) {
    $Arguments.ExePath = $ExePath
}

Write-Host "Running bounded all-variant conditioning sweep"
Write-Host "  n=$($Sizes -join ',')"
Write-Host "  kappa=$($Kappas -join ',')"
Write-Host "  variants=$($Variants -join ',')"
& $SweepScript @Arguments

if (-not $SkipPlots) {
    $PlotScript = Join-Path $PSScriptRoot "plot_conditioning_sweep.py"
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
            throw "Conditioning plot generation failed with exit code $LASTEXITCODE."
        }
    }
}
