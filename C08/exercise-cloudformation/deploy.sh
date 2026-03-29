#!/bin/bash

# Aurora Serverless v2 CloudFormation Deployment Script
# This script helps you deploy, update, and manage your Aurora Serverless stack
#
# ============================================================
# EDUCATIONAL EXERCISE WARNING
# ============================================================
# This script contains REAL AWS CLI commands that would create
# actual AWS resources and incur real charges if executed.
#
# For this exercise, the commands that create, update, or delete
# AWS resources have been COMMENTED OUT and replaced with
# explanations of what each command would do.
#
# Do not uncomment and run these commands unless you:
#   1. Intend to deploy to a live AWS account
#   2. Understand and accept the associated costs
#   3. Have permission to use the AWS account
#
# Estimated cost of a deployed stack: $43–$100+/month
# Always delete the stack when done to stop charges.
# ============================================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
STACK_NAME="${STACK_NAME:-aurora-serverless-stack}"
TEMPLATE_FILE="aurora-serverless.yaml"
PARAMETERS_FILE="parameters.json"
REGION="${AWS_REGION:-us-east-1}"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if AWS CLI is installed
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    print_success "AWS CLI is installed"
}

# Function to validate template
# validate_template sends the template to the CloudFormation API and checks
# for syntax errors and invalid resource configurations. This is a read-only
# call — it does NOT create any resources or incur charges.
validate_template() {
    print_info "Validating CloudFormation template..."

    # This command is safe to run — it only checks syntax, no resources are created.
    if aws cloudformation validate-template \
        --template-body file://${TEMPLATE_FILE} \
        --region ${REGION} > /dev/null 2>&1; then
        print_success "Template is valid"
    else
        print_error "Template validation failed"
        exit 1
    fi
}

# Function to check if stack exists
# describe-stacks is a read-only call — safe to run, no charges.
stack_exists() {
    aws cloudformation describe-stacks \
        --stack-name ${STACK_NAME} \
        --region ${REGION} > /dev/null 2>&1
}

# Function to create stack
# EDUCATIONAL NOTE: This function would submit the template to CloudFormation
# and provision the following real AWS resources:
#   - 1 VPC with 2 private subnets across different Availability Zones
#   - 1 Internet Gateway and route table
#   - 1 Security Group for database access control
#   - 1 RDS DB Subnet Group
#   - 1 Aurora Serverless v2 PostgreSQL cluster (billed per ACU-hour)
#   - 1 Aurora DB Instance (db.serverless class)
#   - 1 Secrets Manager secret (billed per secret per month)
#
# The `wait stack-create-complete` call blocks until all resources are ready,
# which typically takes 5–15 minutes for an Aurora cluster.
create_stack() {
    print_info "Creating stack: ${STACK_NAME}"

    print_warning "============================================================"
    print_warning "EXERCISE MODE: The create-stack command is commented out."
    print_warning "If this were a live deployment, the following would happen:"
    print_warning "  1. CloudFormation receives the template and parameters"
    print_warning "  2. Resources are created in dependency order (VPC first,"
    print_warning "     then subnets, security groups, then Aurora cluster)"
    print_warning "  3. The script waits 5-15 minutes for the cluster to start"
    print_warning "  4. Stack outputs (endpoint, port, secret ARN) are displayed"
    print_warning "  Cost: ~\$43-\$100+/month depending on usage"
    print_warning "============================================================"

    # COMMENTED OUT — would create real AWS resources and incur charges:
    #
    # aws cloudformation create-stack \
    #     --stack-name ${STACK_NAME} \
    #     --template-body file://${TEMPLATE_FILE} \
    #     --parameters file://${PARAMETERS_FILE} \
    #     --region ${REGION} \
    #     --tags Key=Environment,Value=Development Key=ManagedBy,Value=CloudFormation
    #
    # print_info "Stack creation initiated. Waiting for completion..."
    #
    # # This blocks until the stack reaches CREATE_COMPLETE or CREATE_FAILED status
    # aws cloudformation wait stack-create-complete \
    #     --stack-name ${STACK_NAME} \
    #     --region ${REGION}
    #
    # print_success "Stack created successfully!"
}

# Function to update stack
# EDUCATIONAL NOTE: update-stack submits a change set and applies it to an
# existing stack. CloudFormation computes a diff between the current template
# and the new one and only modifies resources that changed.
# Some changes (e.g., engine version, encryption settings) cause a resource
# replacement, which means the old resource is deleted and a new one is created.
# This can cause downtime for database resources.
update_stack() {
    print_info "Updating stack: ${STACK_NAME}"

    print_warning "============================================================"
    print_warning "EXERCISE MODE: The update-stack command is commented out."
    print_warning "If this were a live deployment, the following would happen:"
    print_warning "  1. CloudFormation diffs the current vs. new template"
    print_warning "  2. Only changed resources are modified or replaced"
    print_warning "  3. The script waits for UPDATE_COMPLETE status"
    print_warning "  Note: Some changes require resource replacement (downtime)"
    print_warning "============================================================"

    # COMMENTED OUT — would modify real AWS resources:
    #
    # if aws cloudformation update-stack \
    #     --stack-name ${STACK_NAME} \
    #     --template-body file://${TEMPLATE_FILE} \
    #     --parameters file://${PARAMETERS_FILE} \
    #     --region ${REGION} 2>&1 | grep -q "No updates are to be performed"; then
    #     print_warning "No updates are to be performed"
    #     return 0
    # fi
    #
    # print_info "Stack update initiated. Waiting for completion..."
    #
    # aws cloudformation wait stack-update-complete \
    #     --stack-name ${STACK_NAME} \
    #     --region ${REGION}
    #
    # print_success "Stack updated successfully!"
}

# Function to delete stack
# EDUCATIONAL NOTE: delete-stack removes all resources in the stack in reverse
# dependency order. Because the Aurora cluster has DeletionPolicy: Snapshot,
# CloudFormation first creates a final DB snapshot before deleting the cluster.
# Snapshot storage continues to be billed after deletion — delete snapshots
# you no longer need from the RDS console to fully stop charges.
delete_stack() {
    print_warning "Are you sure you want to delete stack ${STACK_NAME}? (yes/no)"
    read -r response

    if [[ "$response" != "yes" ]]; then
        print_info "Deletion cancelled"
        return 0
    fi

    print_info "Deleting stack: ${STACK_NAME}"

    print_warning "============================================================"
    print_warning "EXERCISE MODE: The delete-stack command is commented out."
    print_warning "If this were a live deployment, the following would happen:"
    print_warning "  1. CloudFormation creates a final Aurora snapshot (billed)"
    print_warning "  2. All stack resources are deleted in reverse order"
    print_warning "  3. Hourly Aurora charges stop once the cluster is deleted"
    print_warning "  4. Remember to delete the final snapshot to stop storage charges"
    print_warning "============================================================"

    # COMMENTED OUT — would delete real AWS resources:
    #
    # aws cloudformation delete-stack \
    #     --stack-name ${STACK_NAME} \
    #     --region ${REGION}
    #
    # print_info "Stack deletion initiated. Waiting for completion..."
    #
    # aws cloudformation wait stack-delete-complete \
    #     --stack-name ${STACK_NAME} \
    #     --region ${REGION}
    #
    # print_success "Stack deleted successfully!"
}

# Function to get stack outputs
# This is a read-only call — safe to run against an existing stack.
get_outputs() {
    print_info "Stack outputs for ${STACK_NAME}:"
    echo ""

    # Read-only — no resources created, no charges
    aws cloudformation describe-stacks \
        --stack-name ${STACK_NAME} \
        --region ${REGION} \
        --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue,Description]' \
        --output table
}

# Function to get stack status
# This is a read-only call — safe to run against an existing stack.
get_status() {
    print_info "Stack status for ${STACK_NAME}:"

    # Read-only — no resources created, no charges
    STATUS=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_NAME} \
        --region ${REGION} \
        --query 'Stacks[0].StackStatus' \
        --output text)

    echo ""
    echo "Status: ${STATUS}"
    echo ""

    # Show recent events
    print_info "Recent stack events:"
    aws cloudformation describe-stack-events \
        --stack-name ${STACK_NAME} \
        --region ${REGION} \
        --max-items 10 \
        --query 'StackEvents[*].[Timestamp,ResourceStatus,ResourceType,LogicalResourceId]' \
        --output table
}

# Function to get connection info
# This is a read-only call — safe to run against an existing stack.
get_connection_info() {
    print_info "Database connection information:"
    echo ""

    # Read-only — no resources created, no charges
    OUTPUTS=$(aws cloudformation describe-stacks \
        --stack-name ${STACK_NAME} \
        --region ${REGION} \
        --query 'Stacks[0].Outputs')

    ENDPOINT=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="ClusterEndpoint") | .OutputValue')
    PORT=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="ClusterPort") | .OutputValue')
    DATABASE=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="DatabaseName") | .OutputValue')

    echo "Endpoint: ${ENDPOINT}"
    echo "Port: ${PORT}"
    echo "Database: ${DATABASE}"
    echo ""
    echo "Connection command:"
    echo "psql -h ${ENDPOINT} -p ${PORT} -U admin -d ${DATABASE}"
}

# Function to display usage
usage() {
    cat << EOF
Aurora Serverless v2 CloudFormation Deployment Script

EDUCATIONAL EXERCISE: Actual deployment commands are commented out.
See the WARNING at the top of this script for details.

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    create      [COMMENTED OUT] Would create a new stack (real AWS resources + charges)
    update      [COMMENTED OUT] Would update an existing stack
    delete      [COMMENTED OUT] Would delete the stack
    status      Show stack status and recent events (read-only, safe)
    outputs     Show stack outputs (read-only, safe)
    connect     Show database connection information (read-only, safe)
    validate    Validate the CloudFormation template (read-only, safe)
    help        Show this help message

Options:
    --stack-name NAME       Name of the CloudFormation stack (default: aurora-serverless-stack)
    --region REGION         AWS region (default: us-east-1)
    --parameters FILE       Parameters file (default: parameters.json)

Environment Variables:
    STACK_NAME              Override default stack name
    AWS_REGION              Override default region

Examples:
    # Validate the template (safe — no resources created)
    $0 validate

    # [EXERCISE ONLY — commented out] Create a new stack
    # $0 create

    # [EXERCISE ONLY — commented out] Update an existing stack
    # $0 update

    # Get stack outputs (safe — read-only)
    $0 outputs

    # Check stack status (safe — read-only)
    $0 status

    # [EXERCISE ONLY — commented out] Delete stack
    # $0 delete

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --stack-name)
            STACK_NAME="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --parameters)
            PARAMETERS_FILE="$2"
            shift 2
            ;;
        create|update|delete|status|outputs|connect|validate|help)
            COMMAND="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    check_aws_cli

    case ${COMMAND} in
        create)
            validate_template
            if stack_exists; then
                print_error "Stack ${STACK_NAME} already exists. Use 'update' to modify it."
                exit 1
            fi
            create_stack
            ;;
        update)
            validate_template
            if ! stack_exists; then
                print_error "Stack ${STACK_NAME} does not exist. Use 'create' to create it."
                exit 1
            fi
            update_stack
            ;;
        delete)
            if ! stack_exists; then
                print_error "Stack ${STACK_NAME} does not exist."
                exit 1
            fi
            delete_stack
            ;;
        status)
            if ! stack_exists; then
                print_error "Stack ${STACK_NAME} does not exist."
                exit 1
            fi
            get_status
            ;;
        outputs)
            if ! stack_exists; then
                print_error "Stack ${STACK_NAME} does not exist."
                exit 1
            fi
            get_outputs
            ;;
        connect)
            if ! stack_exists; then
                print_error "Stack ${STACK_NAME} does not exist."
                exit 1
            fi
            get_connection_info
            ;;
        validate)
            validate_template
            ;;
        help|"")
            usage
            ;;
        *)
            print_error "Unknown command: ${COMMAND}"
            usage
            exit 1
            ;;
    esac
}

main
