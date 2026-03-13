#!/bin/bash

# Aurora Serverless v2 CloudFormation Deployment Script
# This script helps you deploy, update, and manage your Aurora Serverless stack

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
validate_template() {
    print_info "Validating CloudFormation template..."
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
stack_exists() {
    aws cloudformation describe-stacks \
        --stack-name ${STACK_NAME} \
        --region ${REGION} > /dev/null 2>&1
}

# Function to create stack
create_stack() {
    print_info "Creating stack: ${STACK_NAME}"

    aws cloudformation create-stack \
        --stack-name ${STACK_NAME} \
        --template-body file://${TEMPLATE_FILE} \
        --parameters file://${PARAMETERS_FILE} \
        --region ${REGION} \
        --tags Key=Environment,Value=Development Key=ManagedBy,Value=CloudFormation

    print_info "Stack creation initiated. Waiting for completion..."

    aws cloudformation wait stack-create-complete \
        --stack-name ${STACK_NAME} \
        --region ${REGION}

    print_success "Stack created successfully!"
}

# Function to update stack
update_stack() {
    print_info "Updating stack: ${STACK_NAME}"

    if aws cloudformation update-stack \
        --stack-name ${STACK_NAME} \
        --template-body file://${TEMPLATE_FILE} \
        --parameters file://${PARAMETERS_FILE} \
        --region ${REGION} 2>&1 | grep -q "No updates are to be performed"; then
        print_warning "No updates are to be performed"
        return 0
    fi

    print_info "Stack update initiated. Waiting for completion..."

    aws cloudformation wait stack-update-complete \
        --stack-name ${STACK_NAME} \
        --region ${REGION}

    print_success "Stack updated successfully!"
}

# Function to delete stack
delete_stack() {
    print_warning "Are you sure you want to delete stack ${STACK_NAME}? (yes/no)"
    read -r response

    if [[ "$response" != "yes" ]]; then
        print_info "Deletion cancelled"
        return 0
    fi

    print_info "Deleting stack: ${STACK_NAME}"

    aws cloudformation delete-stack \
        --stack-name ${STACK_NAME} \
        --region ${REGION}

    print_info "Stack deletion initiated. Waiting for completion..."

    aws cloudformation wait stack-delete-complete \
        --stack-name ${STACK_NAME} \
        --region ${REGION}

    print_success "Stack deleted successfully!"
}

# Function to get stack outputs
get_outputs() {
    print_info "Stack outputs for ${STACK_NAME}:"
    echo ""

    aws cloudformation describe-stacks \
        --stack-name ${STACK_NAME} \
        --region ${REGION} \
        --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue,Description]' \
        --output table
}

# Function to get stack status
get_status() {
    print_info "Stack status for ${STACK_NAME}:"

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
get_connection_info() {
    print_info "Database connection information:"
    echo ""

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

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    create      Create a new stack
    update      Update an existing stack
    delete      Delete the stack
    status      Show stack status and recent events
    outputs     Show stack outputs
    connect     Show database connection information
    validate    Validate the CloudFormation template
    help        Show this help message

Options:
    --stack-name NAME       Name of the CloudFormation stack (default: aurora-serverless-stack)
    --region REGION         AWS region (default: us-east-1)
    --parameters FILE       Parameters file (default: parameters.json)

Environment Variables:
    STACK_NAME              Override default stack name
    AWS_REGION              Override default region

Examples:
    # Create a new stack
    $0 create

    # Update an existing stack
    $0 update

    # Get stack outputs
    $0 outputs

    # Create stack with custom name
    $0 create --stack-name my-aurora-stack

    # Check stack status
    $0 status

    # Delete stack
    $0 delete

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
            echo ""
            get_outputs
            ;;
        update)
            validate_template
            if ! stack_exists; then
                print_error "Stack ${STACK_NAME} does not exist. Use 'create' to create it."
                exit 1
            fi
            update_stack
            echo ""
            get_outputs
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
