#!/usr/bin/env bash
# =============================================================================
# test_tower.sh — Validate Nextflow Tower / Seqera Platform connectivity
#                 and run the Sequence QC Pipeline with Tower monitoring.
#
# Usage:
#   export TOWER_ACCESS_TOKEN="your_token_here"
#   ./test_tower.sh
#
# Optional environment variables:
#   TOWER_API_ENDPOINT  — Override API endpoint (default: https://api.cloud.seqera.io)
#   TOWER_WORKSPACE_ID  — Numeric workspace ID to target a specific workspace
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

ok()   { echo -e "   ${GREEN}✓${RESET} $*"; }
fail() { echo -e "   ${RED}✗  $*${RESET}"; }
info() { echo -e "   ${CYAN}→${RESET} $*"; }
warn() { echo -e "   ${YELLOW}!${RESET} $*"; }
hdr()  { echo -e "\n${BOLD}$*${RESET}"; }

# ── Configuration ─────────────────────────────────────────────────────────────
TOWER_API_ENDPOINT="${TOWER_API_ENDPOINT:-https://api.cloud.seqera.io}"
# Derive the UI base URL from the API endpoint for convenience links
TOWER_UI_URL="${TOWER_API_ENDPOINT%/api}"
TOWER_UI_URL="${TOWER_UI_URL/api./}"
# If the endpoint is the default, always use the known UI URL
[[ "$TOWER_API_ENDPOINT" == "https://api.cloud.seqera.io" ]] && \
    TOWER_UI_URL="https://cloud.seqera.io"

PASS=0; FAIL=0

# ─────────────────────────────────────────────────────────────────────────────
echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}  Nextflow Tower / Seqera Platform — Connection Test${RESET}"
echo -e "${BOLD}============================================================${RESET}"

# ── Step 1: Check prerequisites ───────────────────────────────────────────────
hdr "1. Checking prerequisites"

# curl
if command -v curl &>/dev/null; then
    ok "curl $(curl --version | head -1 | awk '{print $2}')"
    ((PASS++))
else
    fail "curl is not installed — required for API checks"
    ((FAIL++))
fi

# nextflow
if command -v nextflow &>/dev/null || [[ -x "./nextflow" ]]; then
    NF_CMD="$(command -v nextflow 2>/dev/null || echo './nextflow')"
    ok "Nextflow $($NF_CMD -version 2>&1 | grep -oP 'version \K[\d\.]+' | head -1)"
    ((PASS++))
else
    fail "nextflow not found in PATH and ./nextflow not executable"
    ((FAIL++))
fi

# python3
if command -v python3 &>/dev/null; then
    ok "$(python3 --version)"
    ((PASS++))
else
    fail "python3 not found"
    ((FAIL++))
fi

# ── Step 2: Check TOWER_ACCESS_TOKEN ─────────────────────────────────────────
hdr "2. Checking TOWER_ACCESS_TOKEN"

if [[ -z "${TOWER_ACCESS_TOKEN:-}" ]]; then
    fail "TOWER_ACCESS_TOKEN is not set"
    echo ""
    echo -e "  Set it with:"
    echo -e "  ${CYAN}export TOWER_ACCESS_TOKEN=\"your_token_here\"${RESET}"
    echo -e "  Get a token at: ${TOWER_UI_URL}/tokens"
    echo ""
    echo -e "${RED}Cannot continue without a Tower access token. Exiting.${RESET}"
    exit 1
fi

# Mask token for display
TOKEN_PREVIEW="${TOWER_ACCESS_TOKEN:0:6}...${TOWER_ACCESS_TOKEN: -4}"
ok "TOWER_ACCESS_TOKEN is set  (${TOKEN_PREVIEW})"
((PASS++))

info "API endpoint : ${TOWER_API_ENDPOINT}"
info "UI URL       : ${TOWER_UI_URL}"

if [[ -n "${TOWER_WORKSPACE_ID:-}" ]]; then
    info "Workspace ID : ${TOWER_WORKSPACE_ID}"
fi

# ── Step 3: Test Tower API connectivity ───────────────────────────────────────
hdr "3. Testing Tower API connectivity"

HTTP_STATUS=$(curl -s -o /tmp/tower_api_response.json \
    -w "%{http_code}" \
    -H "Authorization: Bearer ${TOWER_ACCESS_TOKEN}" \
    -H "Accept: application/json" \
    "${TOWER_API_ENDPOINT}/user-info" 2>/dev/null) || HTTP_STATUS="000"

case "$HTTP_STATUS" in
  200)
    ok "API reachable — HTTP 200"
    ((PASS++))

    # Parse user info from the response
    if command -v python3 &>/dev/null; then
        USER_INFO=$(python3 -c "
import json, sys
try:
    d = json.load(open('/tmp/tower_api_response.json'))
    u = d.get('user', d)
    print(f\"  Username : {u.get('userName', u.get('name', 'unknown'))}\" )
    print(f\"  Email    : {u.get('email', 'unknown')}\")
except Exception as e:
    print(f'  (could not parse user info: {e})')
" 2>/dev/null)
        echo -e "${USER_INFO}"
    fi
    ;;
  401)
    fail "HTTP 401 — Invalid or expired access token"
    info "Get a new token at: ${TOWER_UI_URL}/tokens"
    ((FAIL++))
    ;;
  403)
    fail "HTTP 403 — Token lacks required permissions"
    ((FAIL++))
    ;;
  000)
    fail "Could not reach ${TOWER_API_ENDPOINT} — check your network connection"
    ((FAIL++))
    ;;
  *)
    fail "Unexpected HTTP status: ${HTTP_STATUS}"
    ((FAIL++))
    ;;
esac

# ── Step 4: List accessible workspaces (optional) ─────────────────────────────
hdr "4. Listing accessible workspaces"

WS_STATUS=$(curl -s -o /tmp/tower_workspaces.json \
    -w "%{http_code}" \
    -H "Authorization: Bearer ${TOWER_ACCESS_TOKEN}" \
    -H "Accept: application/json" \
    "${TOWER_API_ENDPOINT}/user/workspaces" 2>/dev/null) || WS_STATUS="000"

if [[ "$WS_STATUS" == "200" ]]; then
    ok "Workspace list retrieved"
    ((PASS++))
    if command -v python3 &>/dev/null; then
        python3 -c "
import json
try:
    d = json.load(open('/tmp/tower_workspaces.json'))
    orgs = d.get('orgsAndWorkspaces', [])
    if not orgs:
        print('  (no workspaces found)')
    for item in orgs[:10]:
        ws_id   = item.get('workspaceId', '')
        ws_name = item.get('workspaceName', item.get('orgName', 'unknown'))
        org     = item.get('orgName', '')
        if ws_id:
            print(f'    [{ws_id:>12}]  {org} / {ws_name}')
        else:
            print(f'    (personal)      {ws_name}')
    if len(orgs) > 10:
        print(f'    ... and {len(orgs)-10} more')
except Exception as e:
    print(f'  (could not parse workspace list: {e})')
" 2>/dev/null
    fi
else
    warn "Could not retrieve workspace list (HTTP ${WS_STATUS}) — continuing"
fi

# ── Step 5: Generate test data (if needed) ────────────────────────────────────
hdr "5. Preparing test data"

if [[ ! -f reference.fasta ]]; then
    info "Generating reference genome..."
    python3 generate_reference.py -o reference.fasta -s 42
    ok "reference.fasta created"
else
    ok "reference.fasta already exists"
fi

if [[ ! -f reads.fastq ]]; then
    info "Generating FASTQ reads (5,000 reads for quick test)..."
    python3 generate_fastq.py -o reads.fastq -n 5000 -r reference.fasta -s 42
    ok "reads.fastq created"
else
    ok "reads.fastq already exists"
fi

# ── Step 6: Run pipeline with Tower monitoring ────────────────────────────────
hdr "6. Running pipeline with Tower monitoring"

NF_CMD="${NF_CMD:-nextflow}"

# Build workspace flag if set
WS_FLAG=""
[[ -n "${TOWER_WORKSPACE_ID:-}" ]] && WS_FLAG="-tower-workspace-id ${TOWER_WORKSPACE_ID}"

info "Command: ${NF_CMD} run main.nf -with-tower ${WS_FLAG}"
info "Watch your run at: ${TOWER_UI_URL}/user/runs"
echo ""

set +e
"$NF_CMD" run main.nf \
    -with-tower \
    ${WS_FLAG} \
    --outdir tower_test_results
NF_EXIT=$?
set -e

echo ""
if [[ $NF_EXIT -eq 0 ]]; then
    ok "Pipeline finished successfully"
    ((PASS++))
else
    fail "Pipeline exited with code ${NF_EXIT}"
    ((FAIL++))
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}  Results: ${GREEN}${PASS} passed${RESET}${BOLD}, ${RED}${FAIL} failed${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""
echo -e "  Pipeline outputs : tower_test_results/"
echo -e "  Tower dashboard  : ${TOWER_UI_URL}/user/runs"
echo ""

if [[ $FAIL -gt 0 ]]; then
    echo -e "${RED}One or more checks failed. See output above for details.${RESET}"
    exit 1
fi

exit 0
