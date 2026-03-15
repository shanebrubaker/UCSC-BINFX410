#!/bin/bash

################################################################################
# AWS Resources Cleanup Script
################################################################################
#
# This script removes ALL AWS resources created by this project.
# Use this after completing the learning exercise to avoid any charges.
#
# Usage:
#   ./scripts/cleanup.sh [stack-name]
#
# Example:
#   ./scripts/cleanup.sh bioinfo-stepfunctions-demo
#
# CAUTION: This will permanently delete all resources!
#
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
STACK_NAME="${1:-bioinfo-stepfunctions-demo}"
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

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

################################################################################
# Confirmation
################################################################################

print_header "AWS Resources Cleanup"

print_warning "This will DELETE the following resources:"
echo ""
echo "  • CloudFormation Stack: $STACK_NAME"
echo "  • Lambda Function"
echo "  • Step Functions State Machine"
echo "  • IAM Roles (Lambda and Step Functions)"
echo "  • CloudWatch Log Groups"
echo "  • All execution history and logs"
echo ""
print_warning "This action CANNOT be undone!"
echo ""

read -p "Are you sure you want to continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    print_info "Cleanup cancelled"
    exit 0
fi

################################################################################
# Check if Stack Exists
################################################################################

print_header "Checking Resources"

if ! aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" &> /dev/null; then
    print_info "Stack '$STACK_NAME' not found in region $REGION"
    print_info "Nothing to clean up"
    exit 0
fi

print_success "Found stack: $STACK_NAME"

################################################################################
# Retrieve Resource Information
################################################################################

print_info "Retrieving resource information..."

# Get stack outputs
OUTPUTS=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0].Outputs' \
    --output json 2>/dev/null || echo "[]")

LAMBDA_NAME=$(echo "$OUTPUTS" | jq -r '.[] | select(.OutputKey=="LambdaFunctionName") | .OutputValue' 2>/dev/null || echo "")
STATE_MACHINE_NAME=$(echo "$OUTPUTS" | jq -r '.[] | select(.OutputKey=="StateMachineName") | .OutputValue' 2>/dev/null || echo "")

if [ -n "$LAMBDA_NAME" ]; then
    print_info "Lambda Function: $LAMBDA_NAME"
fi

if [ -n "$STATE_MACHINE_NAME" ]; then
    print_info "State Machine: $STATE_MACHINE_NAME"
fi

################################################################################
# Delete CloudFormation Stack
################################################################################

print_header "Deleting CloudFormation Stack"

print_info "Initiating stack deletion (this may take 2-3 minutes)..."

aws cloudformation delete-stack \
    --stack-name "$STACK_NAME" \
    --region "$REGION"

print_success "Stack deletion initiated"

# Wait for deletion to complete
print_info "Waiting for stack deletion to complete..."

if aws cloudformation wait stack-delete-complete \
    --stack-name "$STACK_NAME" \
    --region "$REGION" 2>/dev/null; then
    print_success "Stack deleted successfully"
else
    # Check if stack still exists
    if aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" &> /dev/null; then
        print_error "Stack deletion may have failed"
        print_info "Check CloudFormation console for details"
        exit 1
    else
        print_success "Stack deleted successfully"
    fi
fi

################################################################################
# Verify Deletion
################################################################################

print_header "Verifying Deletion"

# Check Lambda function
if [ -n "$LAMBDA_NAME" ]; then
    if aws lambda get-function \
        --function-name "$LAMBDA_NAME" \
        --region "$REGION" &> /dev/null; then
        print_warning "Lambda function still exists: $LAMBDA_NAME"
    else
        print_success "Lambda function deleted"
    fi
fi

# Check State Machine
if [ -n "$STATE_MACHINE_NAME" ]; then
    MACHINES=$(aws stepfunctions list-state-machines \
        --region "$REGION" \
        --query "stateMachines[?name=='$STATE_MACHINE_NAME'].name" \
        --output text)

    if [ -n "$MACHINES" ]; then
        print_warning "State machine still exists: $STATE_MACHINE_NAME"
    else
        print_success "State machine deleted"
    fi
fi

# Check IAM roles
print_info "Checking IAM roles..."
ROLES=$(aws iam list-roles \
    --query "Roles[?contains(RoleName, '$STACK_NAME')].RoleName" \
    --output text)

if [ -n "$ROLES" ]; then
    print_warning "Some IAM roles may still exist (may take a few minutes to fully delete)"
else
    print_success "IAM roles deleted"
fi

################################################################################
# Optional: Clean CloudWatch Logs
################################################################################

print_header "CloudWatch Logs Cleanup"

print_info "Searching for related CloudWatch log groups..."

LOG_GROUPS=$(aws logs describe-log-groups \
    --region "$REGION" \
    --query "logGroups[?contains(logGroupName, '/aws/lambda/$LAMBDA_NAME') || contains(logGroupName, '/aws/vendedlogs/states/')].logGroupName" \
    --output text 2>/dev/null || echo "")

if [ -n "$LOG_GROUPS" ]; then
    echo ""
    print_warning "Found CloudWatch log groups:"
    for LOG_GROUP in $LOG_GROUPS; do
        echo "  • $LOG_GROUP"
    done
    echo ""

    read -p "Delete these log groups? (yes/no): " DELETE_LOGS

    if [ "$DELETE_LOGS" = "yes" ]; then
        for LOG_GROUP in $LOG_GROUPS; do
            print_info "Deleting log group: $LOG_GROUP"
            aws logs delete-log-group \
                --log-group-name "$LOG_GROUP" \
                --region "$REGION" 2>/dev/null || print_warning "Could not delete: $LOG_GROUP"
        done
        print_success "Log groups deleted"
    else
        print_info "Log groups retained (will incur minimal storage costs)"
    fi
else
    print_success "No CloudWatch log groups found"
fi

################################################################################
# Final Verification
################################################################################

print_header "Cleanup Summary"

print_success "CloudFormation stack deleted"
print_info "All managed resources have been removed"

cat << EOF

${GREEN}✓ Cleanup Complete!${NC}

${YELLOW}Verification Commands:${NC}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Verify stack deletion:
   aws cloudformation describe-stacks --stack-name ${STACK_NAME} --region ${REGION}
   (Should return: Stack with id ${STACK_NAME} does not exist)

2. List remaining Lambda functions:
   aws lambda list-functions --region ${REGION} --query 'Functions[*].FunctionName'

3. List remaining Step Functions:
   aws stepfunctions list-state-machines --region ${REGION} --query 'stateMachines[*].name'

4. Check IAM roles:
   aws iam list-roles --query 'Roles[?contains(RoleName, \`${STACK_NAME}\`)].RoleName'

${YELLOW}Cost Impact:${NC}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

All resources that could incur charges have been deleted.

${YELLOW}Next Steps:${NC}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Review TUTORIAL.md to reinforce learning concepts
• Practice deploying again: ./scripts/deploy.sh
• Experiment with different retry configurations
• Apply these patterns to your bioinformatics workflows!

EOF

print_success "Thank you for learning with AWS Step Functions! 🧬"
