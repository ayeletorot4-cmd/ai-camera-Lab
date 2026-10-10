# Launcher: pick a video in a file dialog, then run extract_kinematics.py on it.
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()

$here   = Split-Path -Parent $MyInvocation.MyCommand.Path
$parent = Split-Path -Parent $here   # main "AI camera for the lab" folder: shared venv + videos
$python = Join-Path $here 'venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = Join-Path $parent 'venv\Scripts\python.exe' }
$script = Join-Path $here 'extract_kinematics.py'

$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = 'Choose a video to analyze'
$dialog.InitialDirectory = $parent
$dialog.Filter = 'Video files (*.mov;*.mp4;*.avi;*.mkv)|*.mov;*.mp4;*.avi;*.mkv|All files (*.*)|*.*'

$owner = New-Object System.Windows.Forms.Form -Property @{ TopMost = $true }
if ($dialog.ShowDialog($owner) -ne [System.Windows.Forms.DialogResult]::OK) {
    Write-Host 'No video selected - nothing to do.'
    exit 0
}

# Which limb to track?
$limb = ''
while ($limb -eq '') {
    $answer = Read-Host 'Which limb? Type U for upper limb (arm + fingers) or L for lower limb (leg)'
    if ($answer -match '^[Uu]') { $limb = 'upper' }
    elseif ($answer -match '^[Ll]') { $limb = 'lower' }
}
$side = ''
while ($side -eq '') {
    $answer = Read-Host "Which side? Type L for the subject's LEFT or R for the subject's RIGHT"
    if ($answer -match '^[Ll]') { $side = 'left' }
    elseif ($answer -match '^[Rr]') { $side = 'right' }
}
$extraArgs = @('--limb', $limb, '--side', $side)

# Switch events (light start / touching / light off for the 5 switches)
$answer = Read-Host 'Detect switch events? Type Y for yes, or press ENTER to skip'
if ($answer -match '^[Yy]') {
    $extraArgs += '--switches'
    $switchFile = Join-Path $here 'outputs\switch_settings.json'
    $videoName = [System.IO.Path]::GetFileName($dialog.FileName)
    $hasSwitches = $false
    if (Test-Path $switchFile) {
        try { $hasSwitches = $null -ne (Get-Content $switchFile -Raw | ConvertFrom-Json).$videoName } catch {}
    }
    if ($hasSwitches) {
        $answer = Read-Host "Switch positions are saved for $videoName. Press ENTER to reuse them, or type R to mark them again"
        if ($answer -match '^[Rr]') { $extraArgs += '--select-switches' }
    } else {
        Write-Host 'In the video window: drag a box around each switch in order 1 to 5 (RIGHT to LEFT), pressing ENTER after each.'
    }
}

Write-Host ''
Write-Host 'Loading MediaPipe (first frames may take a few seconds)...'
# Quiet MediaPipe/TensorFlow internal log chatter (real errors still show)
$env:GLOG_minloglevel = '2'
$env:TF_CPP_MIN_LOG_LEVEL = '3'
& $python $script $dialog.FileName @extraArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Processing failed (exit code $LASTEXITCODE) - see the message above." -ForegroundColor Red
}
