#!/bin/bash

echo "======================================"
echo "Tower Token Test Script"
echo "======================================"
echo ""

# Step 1: Check if token is set
echo "Step 1: Checking if TOWER_ACCESS_TOKEN is set..."
if [ -z "$TOWER_ACCESS_TOKEN" ]; then
    echo "❌ ERROR: TOWER_ACCESS_TOKEN is NOT set"
    echo ""
    echo "Please run:"
    echo "  export TOWER_ACCESS_TOKEN=<your-token>"
    echo ""
    echo "Get your token from: https://cloud.seqera.io"
    echo "  → Click your profile"
    echo "  → Your tokens"
    echo "  → Copy token"
    exit 1
else
    echo "✅ Token is set"
    echo "   Length: ${#TOWER_ACCESS_TOKEN} characters"
    echo ""
fi

# Step 2: Test authentication
echo "Step 2: Testing authentication with Seqera Platform..."
response=$(curl -s -w "\n%{http_code}" -H "Authorization: Bearer $TOWER_ACCESS_TOKEN" \
  https://cloud.seqera.io/api/user-info)

http_code=$(echo "$response" | tail -1)
body=$(echo "$response" | head -n -1)

if [ "$http_code" -eq 200 ]; then
    echo "✅ Authentication successful!"

    # Try to extract username
    username=$(echo "$body" | grep -o '"userName":"[^"]*"' | cut -d'"' -f4)
    if [ ! -z "$username" ]; then
        echo "   Logged in as: $username"
    fi
    echo ""
else
    echo "❌ Authentication failed (HTTP $http_code)"
    echo "   Response: $body"
    echo ""
    echo "Your token may be invalid or expired."
    echo "Generate a new token at: https://cloud.seqera.io"
    exit 1
fi

# Step 3: Ready to run
echo "======================================"
echo "✅ Everything looks good!"
echo "======================================"
echo ""
echo "You're ready to run with Tower:"
echo "  nextflow run main.nf -with-tower -name 'my-run'"
echo ""
echo "Or run a quick test:"
echo "  nextflow run main.nf -with-tower -resume -name 'tower-test'"
echo ""
