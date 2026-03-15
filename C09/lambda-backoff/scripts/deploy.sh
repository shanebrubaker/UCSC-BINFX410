#!/bin/bash

################################################################################
# Bioinformatics Step Functions Deployment Script
################################################################################
#
# This script deploys the complete AWS infrastructure using CloudFormation.
#
# Prerequisites:
# - AWS CLI installed and configured
# - Valid AWS credentials with permissions to create:
#   * Lambda functions
#   * Step Functions state machines
#   * IAM roles
#   * CloudWatch log groups
#
# Usage:
#   ./scripts/deploy.sh [stack-name]
#
# Example:
#   ./scripts/deploy.sh bioinfo-stepfunctions-demo
#
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
STACK_NAME="${1:-bioinfo-stepfunctions-demo}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
TEMPLATE_FILE="infrastructure/cloudformation.yaml"

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

################################################################################
# Pre-deployment Checks
################################################################################

print_header "Pre-Deployment Checks"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed"
    echo "Please install AWS CLI: https://aws.amazon.com/cli/"
    exit 1
fi
print_success "AWS CLI is installed"

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    print_error "AWS credentials not configured"
    echo "Please run: aws configure"
    exit 1
fi
print_success "AWS credentials are configured"

# Display account information
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
print_info "AWS Account ID: $ACCOUNT_ID"
print_info "Region: $REGION"

# Check if template file exists
if [ ! -f "$TEMPLATE_FILE" ]; then
    print_error "CloudFormation template not found: $TEMPLATE_FILE"
    exit 1
fi
print_success "CloudFormation template found"

# Validate CloudFormation template
print_info "Validating CloudFormation template..."
if aws cloudformation validate-template --template-body file://$TEMPLATE_FILE &> /dev/null; then
    print_success "Template validation passed"
else
    print_error "Template validation failed"
    exit 1
fi

################################################################################
# Deployment
################################################################################

print_header "Deploying Stack: $STACK_NAME"

# Check if stack already exists
if aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" &> /dev/null; then
    print_info "Stack already exists. Updating..."
    OPERATION="update-stack"
    WAIT_CONDITION="stack-update-complete"
else
    print_info "Creating new stack..."
    OPERATION="create-stack"
    WAIT_CONDITION="stack-create-complete"
fi

# Deploy CloudFormation stack
print_info "Deploying CloudFormation stack (this may take 2-3 minutes)..."

aws cloudformation $OPERATION \
    --stack-name "$STACK_NAME" \
    --template-body file://$TEMPLATE_FILE \
    --capabilities CAPABILITY_NAMED_IAM \
    --region "$REGION" \
    --tags \
        Key=Project,Value=BioinformaticsStepFunctions \
        Key=Environment,Value=Learning \
        Key=ManagedBy,Value=Script

# Wait for stack operation to complete
print_info "Waiting for stack operation to complete..."
if aws cloudformation wait $WAIT_CONDITION \
    --stack-name "$STACK_NAME" \
    --region "$REGION"; then
    print_success "Stack deployment completed successfully!"
else
    print_error "Stack deployment failed"
    print_info "Check CloudFormation console for details"
    exit 1
fi

################################################################################
# Retrieve Outputs
################################################################################

print_header "Deployment Information"

# Get stack outputs
OUTPUTS=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0].Outputs' \
    --output json)

# Extract specific outputs
LAMBDA_NAME=$(echo "$OUTPUTS" | jq -r '.[] | select(.OutputKey=="LambdaFunctionName") | .OutputValue')
STATE_MACHINE_ARN=$(echo "$OUTPUTS" | jq -r '.[] | select(.OutputKey=="StateMachineArn") | .OutputValue')
CONSOLE_URL=$(echo "$OUTPUTS" | jq -r '.[] | select(.OutputKey=="ConsoleURL") | .OutputValue')

print_success "Lambda Function: $LAMBDA_NAME"
print_success "State Machine ARN: $STATE_MACHINE_ARN"

################################################################################
# Test Execution
################################################################################

print_header "Testing Deployment"

print_info "Starting test execution of state machine..."

EXECUTION_ARN=$(aws stepfunctions start-execution \
    --state-machine-arn "$STATE_MACHINE_ARN" \
    --input '{"sample_id":"SAMPLE_TEST_12345","attempt":1}' \
    --region "$REGION" \
    --query 'executionArn' \
    --output text)

print_success "Execution started: $EXECUTION_ARN"

# Wait a moment for execution to process
print_info "Waiting 5 seconds for execution to process..."
sleep 5

# Check execution status
EXECUTION_STATUS=$(aws stepfunctions describe-execution \
    --execution-arn "$EXECUTION_ARN" \
    --region "$REGION" \
    --query 'status' \
    --output text)

print_info "Execution Status: $EXECUTION_STATUS"

################################################################################
# Next Steps
################################################################################

print_header "Deployment Complete!"

cat << EOF

${GREEN}✓ All resources deployed successfully!${NC}

${YELLOW}Next Steps:${NC}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. View Step Functions Console:
   ${CONSOLE_URL}

2. Execute the state machine:
   aws stepfunctions start-execution \\
     --state-machine-arn ${STATE_MACHINE_ARN} \\
     --input '{"sample_id":"SAMPLE_12345_RNA_SEQ","attempt":1}'

3. View CloudWatch Logs:
   aws logs tail /aws/lambda/${LAMBDA_NAME} --follow

4. Test with script:
   ./scripts/execute.sh

5. Read the tutorial:
   cat TUTORIAL.md

${YELLOW}Cleanup:${NC}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When finished learning, remove all resources:
   ./scripts/cleanup.sh ${STACK_NAME}

${YELLOW}Cost Information:${NC}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All resources are within AWS Free Tier limits:
- Lambda: 1M free requests/month
- Step Functions: 4,000 free state transitions/month
- CloudWatch Logs: 5GB free ingestion/month

Estimated cost: \$0.00 for typical educational usage

${YELLOW}Support:${NC}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Questions? Check README.md and TUTORIAL.md

EOF

print_success "Setup complete! Happy learning! 🧬"
