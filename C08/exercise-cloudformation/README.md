# Aurora Serverless v2 CloudFormation Template

This CloudFormation template deploys a complete Aurora Serverless v2 PostgreSQL database with all necessary networking infrastructure.

## Architecture Overview

The template creates a fully-functional, production-ready Aurora Serverless v2 database with the following components:

- VPC with private subnets across multiple Availability Zones
- Aurora Serverless v2 PostgreSQL cluster
- Security groups for database access control
- Secrets Manager integration for credential management
- Automated backups and maintenance windows

## Template Components Explained

### 1. Parameters Section

Parameters allow you to customize the deployment without modifying the template.

#### Database Configuration Parameters

**DatabaseName**
- Defines the name of the database to create within the Aurora cluster
- Must start with a letter and contain only alphanumeric characters
- Default: `mydatabase`

**MasterUsername**
- The master/admin username for database access
- Must start with a letter and contain only alphanumeric characters
- Default: `admin`

**MasterUserPassword**
- The master password (stored with `NoEcho: true` for security)
- Minimum 8 characters, alphanumeric only
- This should be changed in production and ideally stored in Secrets Manager

#### Capacity Configuration

**MinCapacity & MaxCapacity**
- Aurora Capacity Units (ACUs) define compute and memory capacity
- Each ACU = approximately 2 GB of memory with corresponding CPU and networking
- Serverless v2 automatically scales between min and max based on workload
- Valid values: 0.5, 1, 1.5, 2, 2.5, 3, 4-10, 16, 32, 64, 128
- Defaults: Min=0.5, Max=2 (cost-effective for development/testing)

#### Network Configuration

**VpcCIDR, PrivateSubnet1CIDR, PrivateSubnet2CIDR**
- Define the IP address ranges for the VPC and subnets
- Defaults create a /16 VPC (65,536 addresses) with two /24 subnets (256 addresses each)
- Subnets must be in different Availability Zones for high availability

#### Backup Configuration

**BackupRetentionPeriod**
- Number of days to retain automated backups (1-35 days)
- Default: 7 days

**PreferredBackupWindow**
- Daily time window for automated backups (UTC)
- Default: 03:00-04:00 (3 AM to 4 AM UTC)

**PreferredMaintenanceWindow**
- Weekly window for system maintenance (UTC)
- Default: Sunday 04:00-05:00 (4 AM to 5 AM UTC)

### 2. Resources Section

This section defines all AWS resources that will be created.

#### VPC Resources

**VPC** (`AWS::EC2::VPC`)
- Creates an isolated virtual network in AWS
- `EnableDnsHostnames` and `EnableDnsSupport` allow DNS resolution for RDS endpoints
- Uses the CIDR block specified in parameters

**InternetGateway** (`AWS::EC2::InternetGateway`)
- Enables communication between VPC and the internet
- Attached to VPC via `VPCGatewayAttachment`
- Useful if you need NAT gateways for updates or external access

**PrivateSubnet1 & PrivateSubnet2** (`AWS::EC2::Subnet`)
- Two private subnets in different Availability Zones
- Required for Aurora multi-AZ deployment (high availability)
- `MapPublicIpOnLaunch: false` ensures instances don't get public IPs
- `!Select [0, !GetAZs '']` automatically selects AZs in the current region

**PrivateRouteTable** (`AWS::EC2::RouteTable`)
- Defines routing rules for the private subnets
- Associated with both private subnets
- No route to Internet Gateway (keeps subnets truly private)

#### Security Resources

**AuroraSecurityGroup** (`AWS::EC2::SecurityGroup`)
- Firewall rules for the Aurora cluster
- **Ingress Rules**: Allows PostgreSQL traffic (port 5432) from within the VPC
- **Egress Rules**: Allows all outbound traffic
- Limits database access to resources within the same VPC

**DBSubnetGroup** (`AWS::RDS::DBSubnetGroup`)
- Groups the private subnets together for Aurora
- Aurora requires subnets in at least 2 Availability Zones
- Ensures high availability and automatic failover capability

#### Database Resources

**AuroraCluster** (`AWS::RDS::DBCluster`)
The main Aurora Serverless v2 cluster resource.

Key Properties:
- **Engine**: `aurora-postgresql` - PostgreSQL-compatible engine
- **EngineMode**: `provisioned` - Required for Serverless v2
- **EngineVersion**: `15.4` - PostgreSQL 15.4 compatible
- **DatabaseName**: Creates initial database with specified name
- **MasterUsername/Password**: Admin credentials
- **DBSubnetGroupName**: Links to the subnet group
- **VpcSecurityGroupIds**: Attaches security group
- **StorageEncrypted**: `true` - Encrypts data at rest using AWS KMS
- **EnableHttpEndpoint**: `true` - Enables Data API for HTTP-based queries
- **ServerlessV2ScalingConfiguration**: Defines min/max capacity for auto-scaling
- **DeletionPolicy/UpdateReplacePolicy**: `Snapshot` - Creates final snapshot before deletion

**AuroraInstance** (`AWS::RDS::DBInstance`)
- The actual database instance within the cluster
- **DBInstanceClass**: `db.serverless` - Indicates Serverless v2 instance
- **PubliclyAccessible**: `false` - Not accessible from internet
- Linked to the cluster via `DBClusterIdentifier`

**DatabaseSecret** (`AWS::SecretsManager::Secret`)
- Stores database credentials securely in AWS Secrets Manager
- Contains username, password, host, port, and cluster identifier as JSON
- Applications can retrieve credentials programmatically
- Better security practice than hardcoding credentials

### 3. Outputs Section

Outputs make important values available after stack creation.

**ClusterEndpoint**
- Write endpoint for the Aurora cluster
- Use this for INSERT, UPDATE, DELETE operations
- Format: `<cluster-name>.<region>.rds.amazonaws.com`

**ClusterReadEndpoint**
- Read-only endpoint for the cluster
- Distributes read traffic across read replicas (if any)
- Use for SELECT queries to reduce load on primary instance

**ClusterPort**
- The port number (default 5432 for PostgreSQL)
- Needed for connection strings

**DatabaseName**
- Echoes back the database name for reference

**ClusterIdentifier**
- The unique identifier for the Aurora cluster
- Useful for AWS CLI/SDK operations

**SecurityGroupId**
- ID of the security group protecting the database
- Use to grant access from EC2 instances, Lambda functions, etc.

**VPCId**
- ID of the created VPC
- Useful for adding additional resources to the same network

**SecretArn**
- ARN of the Secrets Manager secret
- Applications can use this to retrieve credentials securely

**ConnectionString**
- Sample connection string (password masked)
- Format for PostgreSQL connections

## Deployment Instructions

### Prerequisites

- AWS CLI installed and configured
- Appropriate IAM permissions to create VPC, RDS, and Secrets Manager resources
- AWS account with Aurora Serverless v2 available in your region

### Deploy the Stack

#### Option 1: Using AWS CLI

```bash
aws cloudformation create-stack \
  --stack-name aurora-serverless-stack \
  --template-body file://aurora-serverless.yaml \
  --parameters file://parameters.json \
  --capabilities CAPABILITY_IAM
```

#### Option 2: Using AWS Console

1. Navigate to CloudFormation in AWS Console
2. Click "Create Stack" > "With new resources"
3. Upload the `aurora-serverless.yaml` template
4. Enter parameter values
5. Review and create

### Monitor Deployment

```bash
aws cloudformation describe-stacks \
  --stack-name aurora-serverless-stack \
  --query 'Stacks[0].StackStatus'
```

### Get Connection Information

```bash
aws cloudformation describe-stacks \
  --stack-name aurora-serverless-stack \
  --query 'Stacks[0].Outputs'
```

## Connecting to the Database

### From EC2 Instance in Same VPC

```bash
psql -h <ClusterEndpoint> -U admin -d mydatabase
```

### From Lambda Function

Add the Lambda function to the same VPC and security group:

```python
import psycopg2
import boto3
import json

def get_secret():
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='<SecretArn>')
    return json.loads(response['SecretString'])

def lambda_handler(event, context):
    secret = get_secret()
    conn = psycopg2.connect(
        host=secret['host'],
        port=secret['port'],
        user=secret['username'],
        password=secret['password'],
        database='mydatabase'
    )
    # Your database operations here
```

### Using Data API (No VPC Required)

Enable the Data API for HTTP-based access without managing connections:

```python
import boto3

rds_data = boto3.client('rds-data')

response = rds_data.execute_statement(
    resourceArn='<ClusterArn>',
    secretArn='<SecretArn>',
    database='mydatabase',
    sql='SELECT * FROM users LIMIT 10'
)
```

## Cost Considerations

### Aurora Serverless v2 Pricing

- Billed per ACU-hour (Aurora Capacity Unit per hour)
- Scales automatically between MinCapacity and MaxCapacity
- No charge when database is paused (if MinCapacity = 0.5)
- Storage billed per GB-month
- I/O operations billed per million requests
- Backup storage beyond retention period billed per GB-month

### Example Cost Calculation

With default settings (Min: 0.5 ACU, Max: 2 ACU):
- If database runs at 0.5 ACU continuously: ~$43/month
- If database scales to 2 ACU for 8 hours/day: ~$90/month
- Storage (100 GB): ~$10/month
- Backup storage (50 GB beyond retention): ~$5/month

Check current pricing: https://aws.amazon.com/rds/aurora/pricing/

## Security Best Practices

1. **Change Default Password**: Update `MasterUserPassword` in parameters.json
2. **Use Secrets Manager**: Retrieve credentials from Secrets Manager instead of hardcoding
3. **Restrict Security Group**: Limit ingress to specific security groups, not entire VPC CIDR
4. **Enable Encryption**: Template enables encryption at rest (can add encryption in transit)
5. **Regular Backups**: Default 7-day retention; increase for production
6. **Monitor Access**: Enable CloudWatch Logs and Performance Insights
7. **Rotate Credentials**: Implement automatic password rotation via Secrets Manager

## Scaling Behavior

Aurora Serverless v2 automatically scales:

- **Scale Up**: When CPU > 75% or connections increase
- **Scale Down**: When resources are underutilized for sustained period
- **Scaling Time**: Typically scales in seconds
- **No Downtime**: Scaling happens without connection drops
- **Granular Steps**: Scales in 0.5 ACU increments

## Updating the Stack

To modify the stack:

```bash
aws cloudformation update-stack \
  --stack-name aurora-serverless-stack \
  --template-body file://aurora-serverless.yaml \
  --parameters file://parameters.json
```

Some changes may cause replacement (downtime):
- Changing engine version (major version)
- Changing encryption settings
- Changing DB cluster identifier

## Deleting the Stack

```bash
aws cloudformation delete-stack --stack-name aurora-serverless-stack
```

A final snapshot will be created automatically (due to `DeletionPolicy: Snapshot`).

## Troubleshooting

### Stack Creation Fails

- Check IAM permissions
- Verify subnet CIDR blocks don't overlap
- Ensure Aurora Serverless v2 is available in your region
- Check CloudFormation Events tab for specific errors

### Cannot Connect to Database

- Verify security group allows traffic from your source
- Confirm you're connecting from within the VPC
- Check connection string (endpoint, port, database name)
- Verify credentials are correct

### Slow Performance

- Check current ACU usage in CloudWatch
- Consider increasing MaxCapacity
- Review query performance and indexes
- Enable Performance Insights for detailed metrics

## Additional Features to Consider

1. **Enhanced Monitoring**: Add `MonitoringInterval` and `MonitoringRoleArn`
2. **Performance Insights**: Add `EnablePerformanceInsights: true`
3. **CloudWatch Logs**: Export PostgreSQL logs to CloudWatch
4. **Read Replicas**: Add additional DB instances for read scaling
5. **Global Database**: Set up cross-region replication
6. **IAM Authentication**: Enable IAM database authentication
7. **Parameter Groups**: Customize database parameters
8. **VPC Endpoints**: Add VPC endpoints for private AWS service access

## References

- [Aurora Serverless v2 Documentation](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/aurora-serverless-v2.html)
- [CloudFormation RDS Resources](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/AWS_RDS.html)
- [Aurora PostgreSQL Versions](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraPostgreSQLReleaseNotes/Welcome.html)
- [Best Practices for Aurora](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/Aurora.BestPractices.html)
