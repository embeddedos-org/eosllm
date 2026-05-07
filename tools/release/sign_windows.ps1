# tools/release/sign_windows.ps1
#
# Authenticode-sign every .exe in a staging directory using a PFX cert
# pulled from base64-encoded environment variables. Called from
# .github/workflows/release.yml::build-windows-x86_64.
#
# Required env (each is one of GitHub Actions secrets.*):
#   WINDOWS_CERT_PFX_B64    base64 of the .pfx
#   WINDOWS_CERT_PASSWORD   passphrase for the .pfx
#
# Usage:
#     pwsh -File tools/release/sign_windows.ps1 -StagingDir <dir>
#
# After successful run every .exe in <dir> is signed with SHA-256 and
# timestamped via the DigiCert RFC 3161 server.

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)] [string] $StagingDir
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $StagingDir)) {
    throw "StagingDir not found: $StagingDir"
}

$pfxB64 = $env:WINDOWS_CERT_PFX_B64
$pfxPwd = $env:WINDOWS_CERT_PASSWORD
if ([string]::IsNullOrWhiteSpace($pfxB64)) { throw "WINDOWS_CERT_PFX_B64 is empty" }
if ([string]::IsNullOrWhiteSpace($pfxPwd)) { throw "WINDOWS_CERT_PASSWORD is empty" }

# Decode the certificate to a temp .pfx.
$pfxPath = Join-Path $env:RUNNER_TEMP "eosllm-codesign.pfx"
[System.IO.File]::WriteAllBytes($pfxPath, [System.Convert]::FromBase64String($pfxB64))

# Locate signtool.exe. The Windows runner ships the Win10 SDK in
# C:\Program Files (x86)\Windows Kits\10\bin\<ver>\x64\signtool.exe.
$kit = Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin\*\x64\signtool.exe" `
    | Sort-Object LastWriteTime -Descending `
    | Select-Object -First 1
if (-not $kit) { throw "signtool.exe not found under Windows Kits\10" }
$signtool = $kit.FullName

# Sign each .exe. /fd sha256 + /td sha256 + /tr (RFC3161 timestamp)
# is the modern Microsoft-recommended invocation; SHA-1 is deprecated.
Get-ChildItem $StagingDir -Recurse -Include *.exe | ForEach-Object {
    Write-Host "Signing: $($_.FullName)"
    & $signtool sign `
        /f $pfxPath `
        /p $pfxPwd `
        /fd sha256 `
        /tr "http://timestamp.digicert.com" `
        /td sha256 `
        $_.FullName
    if ($LASTEXITCODE -ne 0) { throw "signtool failed: $LASTEXITCODE on $($_.FullName)" }

    # Verify the signature stuck.
    & $signtool verify /pa /v $_.FullName
    if ($LASTEXITCODE -ne 0) { throw "signtool verify failed: $LASTEXITCODE on $($_.FullName)" }
}

# Clean up.
Remove-Item -Force $pfxPath
Write-Host "sign_windows: OK"
