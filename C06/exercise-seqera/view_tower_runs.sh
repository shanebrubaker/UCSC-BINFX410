#!/bin/bash

echo "======================================"
echo "Finding Your Tower Runs"
echo "======================================"
echo ""

if [ -z "$TOWER_ACCESS_TOKEN" ]; then
    echo "❌ TOWER_ACCESS_TOKEN not set"
    echo "Run: export TOWER_ACCESS_TOKEN=<your-token>"
    exit 1
fi

echo "🔍 Fetching your recent runs..."
echo ""

# Get user info
response=$(curl -s -H "Authorization: Bearer $TOWER_ACCESS_TOKEN" \
  https://cloud.seqera.io/api/user-info)

username=$(echo "$response" | grep -o '"userName":"[^"]*"' | cut -d'"' -f4)
userId=$(echo "$response" | grep -o '"id":[0-9]*' | head -1 | cut -d':' -f2)

if [ ! -z "$username" ]; then
    echo "✅ Logged in as: $username"
    echo ""
fi

# Get workflows
echo "📊 Your recent workflows:"
echo ""

workflows=$(curl -s -H "Authorization: Bearer $TOWER_ACCESS_TOKEN" \
  "https://cloud.seqera.io/api/workflow?max=10")

# Extract and display workflow info
echo "$workflows" | grep -o '"runName":"[^"]*"' | cut -d'"' -f4 | nl

echo ""
echo "======================================"
echo "View all runs at:"
echo "https://cloud.seqera.io/user/$username/watch"
echo "======================================"
echo ""
