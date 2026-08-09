#!/usr/bin/env bash
# Smoke test: sends all 10 challenge scenarios and asserts 200 + non-empty answer.
# Usage: bash scripts/smoke_test.sh [BASE_URL]
set -euo pipefail

BASE_URL="${1:-http://localhost:8080}"
PASS=0
FAIL=0

check() {
  local msg="$1"
  local user="${2:-cliente1988}"
  local payload="{\"message\": $(echo "$msg" | python3 -c "import json,sys; print(json.dumps(sys.stdin.read().strip()))"), \"user_id\": \"$user\"}"

  local http_code answer
  http_code=$(curl -s -o /tmp/smoke_resp.json -w "%{http_code}" \
    -X POST "$BASE_URL/chat" \
    -H "Content-Type: application/json" \
    -d "$payload")
  answer=$(python3 -c "import json; d=json.load(open('/tmp/smoke_resp.json')); print(d.get('answer',''))" 2>/dev/null || echo "")

  if [[ "$http_code" == "200" && -n "$answer" ]]; then
    echo "✅  [$http_code] ${msg:0:60}..."
    PASS=$((PASS + 1))
  else
    echo "❌  [$http_code] ${msg:0:60}..."
    cat /tmp/smoke_resp.json
    FAIL=$((FAIL + 1))
  fi
}

echo "=== Smoke test → $BASE_URL ==="
echo ""

check "What's the difference between the Get Clássica and the Get Smart?"
check "What's the weather forecast in Porto Alegre tomorrow?"
check "When will the money from yesterday's sales be deposited?"
check "Do I need a bank account to receive my sales via Pix?"
check "My card machine won't connect to the internet, what should I do?"
check "How does receivables advance (antecipação) work with Getnet?"
check "What's the euro exchange rate today?"
check "My card machine is showing a transaction decline error."
check "How many installments can I split a sale into with the crediário?"
check "Can I sell through WhatsApp using the Payment Link?"

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[[ $FAIL -eq 0 ]] || exit 1
