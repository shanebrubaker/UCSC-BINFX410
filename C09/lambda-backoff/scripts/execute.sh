#!/bin/bash

################################################################################
# Step Functions Execution Script
################################################################################
#
# This script executes the Step Functions state machine and monitors the results.
#
# Usage:
#   ./scripts/execute.sh [stack-name] [sample-id]
#
# Example:
#   ./scripts/execute.sh bioinfo-stepfunctions-demo SAMPLE_RNA_SEQ_001
#
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
STACK_NAME="${1:-bioinfo-stepfunctions-demo}"
SAMPLE_ID="${2:-SAMPLE_$(date +%s)_RNA_SEQ}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_data() {
    echo -e "${CYAN}$1${NC}"
}

################################################################################
# Get State Machine ARN
################################################################################

print_header "Executing Step Functions State Machine"

print_info "Retrieving state machine ARN from stack: $STACK_NAME"

STATE_MACHINE_ARN=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`StateMachineArn`].OutputValue' \
    --output text 2>/dev/null)

if [ -z "$STATE_MACHINE_ARN" ]; then
    print_error "Could not find state machine ARN"
    print_info "Make sure the stack is deployed: ./scripts/deploy.sh"
    exit 1
fi

print_success "State Machine ARN: $STATE_MACHINE_ARN"

################################################################################
# Start Execution
################################################################################

print_header "Starting Execution"

print_info "Sample ID: $SAMPLE_ID"
print_info "Starting execution..."

EXECUTION_ARN=$(aws stepfunctions start-execution \
    --state-machine-arn "$STATE_MACHINE_ARN" \
    --name "execution-$(date +%s)" \
    --input "{\"sample_id\":\"$SAMPLE_ID\",\"attempt\":1}" \
    --region "$REGION" \
    --query 'executionArn' \
    --output text)

print_success "Execution started!"
print_data "Execution ARN: $EXECUTION_ARN"

################################################################################
# Monitor Execution
################################################################################

print_header "Monitoring Execution"

print_info "Polling execution status (this may take up to 60 seconds)..."
echo ""

# Monitor execution with timeout
MAX_WAIT=90
ELAPSED=0
INTERVAL=3

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Get execution status
    EXECUTION_INFO=$(aws stepfunctions describe-execution \
        --execution-arn "$EXECUTION_ARN" \
        --region "$REGION" \
        --output json)

    STATUS=$(echo "$EXECUTION_INFO" | jq -r '.status')

    # Print status with timestamp
    TIMESTAMP=$(date '+%H:%M:%S')

    if [ "$STATUS" = "RUNNING" ]; then
        echo -e "${YELLOW}[$TIMESTAMP]${NC} Status: ${YELLOW}RUNNING${NC} (waiting...)"
    elif [ "$STATUS" = "SUCCEEDED" ]; then
        echo -e "${GREEN}[$TIMESTAMP]${NC} Status: ${GREEN}SUCCEEDED${NC}"
        break
    elif [ "$STATUS" = "FAILED" ]; then
        echo -e "${RED}[$TIMESTAMP]${NC} Status: ${RED}FAILED${NC}"
        break
    elif [ "$STATUS" = "TIMED_OUT" ]; then
        echo -e "${RED}[$TIMESTAMP]${NC} Status: ${RED}TIMED_OUT${NC}"
        break
    elif [ "$STATUS" = "ABORTED" ]; then
        echo -e "${RED}[$TIMESTAMP]${NC} Status: ${RED}ABORTED${NC}"
        break
    fi

    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo ""

################################################################################
# Display Results
################################################################################

print_header "Execution Results"

# Get final execution details
EXECUTION_INFO=$(aws stepfunctions describe-execution \
    --execution-arn "$EXECUTION_ARN" \
    --region "$REGION" \
    --output json)

STATUS=$(echo "$EXECUTION_INFO" | jq -r '.status')
START_DATE=$(echo "$EXECUTION_INFO" | jq -r '.startDate')
STOP_DATE=$(echo "$EXECUTION_INFO" | jq -r '.stopDate')

print_info "Status: $STATUS"
print_info "Started: $START_DATE"
print_info "Stopped: $STOP_DATE"

# Display output or error
if [ "$STATUS" = "SUCCEEDED" ]; then
    OUTPUT=$(echo "$EXECUTION_INFO" | jq -r '.output')
    print_success "Execution completed successfully!"
    echo ""
    print_data "Output:"
    echo "$OUTPUT" | jq '.'

elif [ "$STATUS" = "FAILED" ]; then
    print_error "Execution failed after all retry attempts"
    if echo "$EXECUTION_INFO" | jq -e '.error' > /dev/null 2>&1; then
        ERROR=$(echo "$EXECUTION_INFO" | jq -r '.error')
        CAUSE=$(echo "$EXECUTION_INFO" | jq -r '.cause')
        print_data "Error: $ERROR"
        print_data "Cause: $CAUSE"
    fi
fi

################################################################################
# Display Execution History
################################################################################

print_header "Execution History (Retry Attempts)"

print_info "Fetching execution history to show retry attempts..."
echo ""

HISTORY=$(aws stepfunctions get-execution-history \
    --execution-arn "$EXECUTION_ARN" \
    --region "$REGION" \
    --output json)

# Extract Lambda invocation attempts
echo "$HISTORY" | jq -r '
.events[] |
select(.type == "TaskStateEntered" or .type == "TaskFailed" or .type == "TaskSucceeded") |
"\(.timestamp) | \(.type) | \(if .taskFailedEventDetails then .taskFailedEventDetails.error else "N/A" end)"
' | while IFS='|' read -r timestamp event_type error; do
    timestamp=$(echo "$timestamp" | xargs)
    event_type=$(echo "$event_type" | xargs)
    error=$(echo "$error" | xargs)

    if [ "$event_type" = "TaskStateEntered" ]; then
        echo -e "${CYAN}[$timestamp]${NC} Attempting Lambda invocation..."
    elif [ "$event_type" = "TaskFailed" ]; then
        echo -e "${RED}[$timestamp]${NC} Attempt failed: $error"
    elif [ "$event_type" = "TaskSucceeded" ]; then
        echo -e "${GREEN}[$timestamp]${NC} Attempt succeeded!"
    fi
done

################################################################################
# CloudWatch Logs
################################################################################

print_header "CloudWatch Logs"

LAMBDA_NAME=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`LambdaFunctionName`].OutputValue' \
    --output text 2>/dev/null)

LOG_GROUP="/aws/lambda/$LAMBDA_NAME"

print_info "Recent Lambda logs (last 10 minutes):"
echo ""

aws logs tail "$LOG_GROUP" \
    --since 10m \
    --format short \
    --region "$REGION" 2>/dev/null || print_info "No recent logs found"

################################################################################
# Summary
################################################################################

print_header "Summary"

cat << EOF

${YELLOW}View Details:${NC}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Step Functions Console:
   https://console.aws.amazon.com/states/home?region=${REGION}#/executions/details/${EXECUTION_ARN}

2. View all CloudWatch logs:
   aws logs tail ${LOG_GROUP} --follow

3. Run another execution:
   ./scripts/execute.sh ${STACK_NAME} SAMPLE_NEW_001

${YELLOW}Learning Points:${NC}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Notice the time intervals between retry attempts (exponential backoff)
- Observe how failures are automatically retried up to 5 times
- Check CloudWatch logs to see detailed attempt information
- Read TUTORIAL.md for more about retry patterns in bioinformatics

EOF

print_success "Execution monitoring complete!"
