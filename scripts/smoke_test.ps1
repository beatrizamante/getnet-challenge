# Smoke test for Windows PowerShell.
# Usage: .\scripts\smoke_test.ps1 [-BaseUrl http://localhost:8080]
param([string]$BaseUrl = "http://localhost:8080")

$scenarios = @(
    "What's the difference between the Get Clássica and the Get Smart?",
    "What's the weather forecast in Porto Alegre tomorrow?",
    "When will the money from yesterday's sales be deposited?",
    "Do I need a bank account to receive my sales via Pix?",
    "My card machine won't connect to the internet, what should I do?",
    "How does receivables advance (antecipação) work with Getnet?",
    "What's the euro exchange rate today?",
    "My card machine is showing a transaction decline error.",
    "How many installments can I split a sale into with the crediário?",
    "Can I sell through WhatsApp using the Payment Link?"
)

$pass = 0; $fail = 0

Write-Host "=== Smoke test -> $BaseUrl ===" -ForegroundColor Cyan

foreach ($msg in $scenarios) {
    $body = @{ message = $msg; user_id = "cliente1988" } | ConvertTo-Json -Compress
    try {
        $resp = Invoke-RestMethod -Uri "$BaseUrl/chat" -Method POST `
            -ContentType "application/json" -Body $body -ErrorAction Stop
        if ($resp.answer -and $resp.source_agent) {
            Write-Host "OK  $($msg.Substring(0, [Math]::Min(60, $msg.Length)))..." -ForegroundColor Green
            $pass++
        } else {
            Write-Host "FAIL  $($msg.Substring(0, [Math]::Min(60, $msg.Length)))..." -ForegroundColor Red
            $fail++
        }
    } catch {
        Write-Host "FAIL [$($_.Exception.Response.StatusCode)]  $($msg.Substring(0, [Math]::Min(60, $msg.Length)))..." -ForegroundColor Red
        $fail++
    }
}

Write-Host "`n=== Results: $pass passed, $fail failed ===" -ForegroundColor Cyan
if ($fail -gt 0) { exit 1 }
