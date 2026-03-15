# AWS Step Functions Educational Project: Exponential Backoff Retry Pattern

## 🧬 For Bioinformatics Students Learning Cloud Orchestration

This educational project demonstrates **exponential backoff retry logic** using AWS Step Functions and Lambda, a critical pattern for building resilient bioinformatics pipelines that interact with unreliable external APIs.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Learning Objectives](#learning-objectives)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Deployment Options](#deployment-options)
- [Usage Examples](#usage-examples)
- [Cost Information](#cost-information)
- [Troubleshooting](#troubleshooting)
- [Learning Resources](#learning-resources)
- [Cleanup](#cleanup)

---

## 🎯 Overview

This project simulates a common bioinformatics scenario: **calling an unreliable external API** (such as NCBI, UniProt, or UCSC Genome Browser) that may fail due to:
- Rate limiting (HTTP 429)
- Temporary service unavailability (HTTP 503)
- Network timeouts
- Internal server errors (HTTP 500)

You'll learn how AWS Step Functions automatically handles these failures using **exponential backoff retry logic**, preventing overwhelming the failing service while maximizing success rates.

### Why This Matters in Bioinformatics

Real-world genomics workflows often involve:
- Fetching sequences from NCBI Entrez API
- Querying protein databases (UniProt, PDB)
- Accessing cloud-based analysis services
- Retrieving reference genomes from UCSC

These APIs have **rate limits** and can experience **transient failures**. Without proper retry logic, your entire pipeline could fail due to a single temporary network issue.

---

## 🎓 Learning Objectives

By completing this project, you will:

1. **Understand exponential backoff** and its mathematical properties
2. **Deploy AWS Step Functions** with retry policies
3. **Create Lambda functions** with proper error handling
4. **Monitor executions** using CloudWatch Logs
5. **Implement Infrastructure as Code** with CloudFormation
6. **Apply cloud patterns** to bioinformatics workflows

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Step Functions                           │
│                  State Machine with                         │
│                 Exponential Backoff                         │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Retry Configuration:                                │   │
│  │ • IntervalSeconds: 2                                │   │
│  │ • MaxAttempts: 5                                    │   │
│  │ • BackoffRate: 2.0                                  │   │
│  │                                                     │   │
│  │ Timeline:                                           │   │
│  │   Attempt 1: Immediate                              │   │
│  │   Attempt 2: Wait 2s                                │   │
│  │   Attempt 3: Wait 4s                                │   │
│  │   Attempt 4: Wait 8s                                │   │
│  │   Attempt 5: Wait 16s                               │   │
│  │   Attempt 6: Wait 32s                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           │ Invokes (with retries)          │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Lambda Function                           │   │
│  │    Unreliable API Simulator                         │   │
│  │    (60% failure rate)                               │   │
│  │                                                     │   │
│  │  Simulates:                                         │   │
│  │  • HTTP 503 (Service Unavailable)                   │   │
│  │  • HTTP 429 (Rate Limit Exceeded)                   │   │
│  │  • HTTP 500 (Internal Server Error)                 │   │
│  │  • Network timeouts                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           │ Logs                            │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │          CloudWatch Logs                            │   │
│  │    • Execution history                              │   │
│  │    • Retry attempts                                 │   │
│  │    • Timestamps                                     │   │
│  │    • Error details                                  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ Prerequisites

### Required

1. **AWS Account** (free tier eligible)
   - Sign up at: https://aws.amazon.com/free/

2. **AWS CLI** installed and configured
   ```bash
   # Install AWS CLI
   # macOS
   brew install awscli

   # Linux
   pip install awscli

   # Windows
   # Download from: https://aws.amazon.com/cli/

   # Configure credentials
   aws configure
   ```

3. **jq** (JSON processor for scripts)
   ```bash
   # macOS
   brew install jq

   # Linux
   sudo apt-get install jq

   # Windows
   # Download from: https://stedolan.github.io/jq/
   ```

### Recommended

- Basic understanding of Python
- Familiarity with command-line interfaces
- AWS Console access for visualization

---

## 🚀 Quick Start

### 1. Clone or Download This Project

```bash
cd /path/to/your/workspace
# If using git
git clone <repository-url>
cd bioinfo-stepfunctions-demo

# Or download and extract the ZIP file
```

### 2. Make Scripts Executable

```bash
chmod +x scripts/*.sh
```

### 3. Deploy the Infrastructure

```bash
./scripts/deploy.sh
```

This will:
- Validate your AWS credentials
- Create CloudFormation stack
- Deploy Lambda function
- Create Step Functions state machine
- Configure IAM roles
- Set up CloudWatch logging

**Expected time:** 2-3 minutes

### 4. Execute the State Machine

```bash
./scripts/execute.sh
```

This will:
- Start a new execution
- Monitor progress in real-time
- Display retry attempts
- Show final results

### 5. View Results

Open the AWS Console URL provided in the output to see:
- Visual workflow execution
- Retry timeline
- CloudWatch logs

### 6. Clean Up (When Finished)

```bash
./scripts/cleanup.sh
```

This removes ALL resources to avoid any charges.

---

## 📁 Project Structure

```
.
├── README.md                           # This file
├── TUTORIAL.md                         # Detailed learning guide
│
├── lambda/
│   └── unreliable_api_simulator.py     # Lambda function (Python 3.12)
│
├── infrastructure/
│   ├── cloudformation.yaml             # Complete IaC template
│   ├── state-machine-definition.json   # Production state machine
│   └── state-machine-definition-commented.json  # Educational version
│
├── scripts/
│   ├── deploy.sh                       # Deployment automation
│   ├── execute.sh                      # Execution and monitoring
│   └── cleanup.sh                      # Resource cleanup
│
└── examples/
    └── sample-output.txt               # Expected execution output
```

---

## 🔧 Deployment Options

### Option 1: Automated Script (Recommended)

```bash
./scripts/deploy.sh [stack-name]
```

**Pros:** Simple, automated, includes validation
**Best for:** Most students

### Option 2: Manual CloudFormation

```bash
aws cloudformation create-stack \
  --stack-name bioinfo-stepfunctions-demo \
  --template-body file://infrastructure/cloudformation.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-east-1
```

**Pros:** More control, good for learning AWS CLI
**Best for:** Students comfortable with AWS CLI

### Option 3: AWS Console

1. Go to CloudFormation console
2. Create Stack → Upload template
3. Select `infrastructure/cloudformation.yaml`
4. Follow wizard

**Pros:** Visual interface
**Best for:** AWS Console learners

---

## 💡 Usage Examples

### Execute with Custom Sample ID

```bash
./scripts/execute.sh bioinfo-stepfunctions-demo SAMPLE_RNA_SEQ_001
```

### View CloudWatch Logs Live

```bash
aws logs tail /aws/lambda/bioinfo-stepfunctions-demo-unreliable-api --follow
```

### Execute via AWS CLI

```bash
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:ACCOUNT_ID:stateMachine:bioinfo-stepfunctions-demo-retry-demo \
  --input '{"sample_id":"SAMPLE_12345","attempt":1}'
```

### Test Lambda Locally

```bash
cd lambda
python3 unreliable_api_simulator.py
```

### View Execution History

```bash
aws stepfunctions describe-execution \
  --execution-arn arn:aws:states:us-east-1:ACCOUNT_ID:execution:bioinfo-stepfunctions-demo-retry-demo:exec-name
```

---

## 💰 Cost Information

### Free Tier Limits (Monthly)

| Service | Free Tier | This Project Usage | Cost |
|---------|-----------|-------------------|------|
| **Lambda** | 1M requests<br>400,000 GB-seconds | ~100 invocations<br>~0.3 GB-seconds | $0.00 |
| **Step Functions** | 4,000 state transitions | ~50 transitions | $0.00 |
| **CloudWatch Logs** | 5GB ingestion<br>5GB storage | ~10MB | $0.00 |

**Total estimated cost:** $0.00 (well within free tier)

### Beyond Free Tier

If you exceed free tier limits:
- Lambda: $0.20 per 1M requests + $0.0000166667 per GB-second
- Step Functions: $0.025 per 1,000 state transitions
- CloudWatch Logs: $0.50 per GB ingested

**For typical educational usage:** You would need to run this project thousands of times to incur any charges.

### Cost Safety Measures

1. **7-day log retention** (minimal storage)
2. **128MB Lambda memory** (lowest cost tier)
3. **Simple state machine** (<100 transitions per execution)
4. **Cleanup script** removes all resources

---

## 🐛 Troubleshooting

### Issue: "Stack already exists"

**Solution:** Update existing stack or use different name
```bash
./scripts/deploy.sh bioinfo-stepfunctions-demo-v2
```

### Issue: "Access Denied" errors

**Cause:** Insufficient IAM permissions

**Solution:** Ensure your AWS user has permissions for:
- CloudFormation (full access)
- Lambda (full access)
- Step Functions (full access)
- IAM (role creation)
- CloudWatch Logs (write access)

### Issue: Execution shows "RUNNING" forever

**Cause:** Lambda function might be stuck

**Solution:**
1. Check CloudWatch logs for errors
2. Verify Lambda timeout (3 seconds)
3. Ensure Lambda has proper IAM role

### Issue: All executions succeed immediately

**Cause:** Lambda failure rate might be too low (random)

**Solution:** Run multiple executions - 60% failure rate means ~40% succeed on first try

### Issue: jq command not found

**Solution:** Install jq
```bash
# macOS
brew install jq

# Linux
sudo apt-get install jq
```

### Issue: Cannot delete stack - resources in use

**Cause:** Active executions or log streams

**Solution:**
1. Wait a few minutes for executions to complete
2. Manually stop any running executions
3. Try cleanup script again

---

## 📚 Learning Resources

### Included in This Project

1. **TUTORIAL.md** - Comprehensive learning guide covering:
   - Exponential backoff mathematics
   - Step Functions concepts
   - Real-world bioinformatics applications
   - CloudWatch log analysis

2. **Commented Code** - All files include detailed educational comments

3. **Sample Output** - `examples/sample-output.txt` shows expected results

### External Resources

**AWS Documentation:**
- [Step Functions Developer Guide](https://docs.aws.amazon.com/step-functions/latest/dg/)
- [Lambda Developer Guide](https://docs.aws.amazon.com/lambda/latest/dg/)
- [CloudWatch Logs](https://docs.aws.amazon.com/cloudwatch/latest/logs/)

**Bioinformatics APIs:**
- [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/)
- [UniProt REST API](https://www.uniprot.org/help/api)
- [UCSC Genome Browser API](https://genome.ucsc.edu/goldenPath/help/api.html)

**Retry Patterns:**
- [AWS Architecture Blog - Retry Strategies](https://aws.amazon.com/blogs/architecture/)
- [Exponential Backoff and Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)

---

## 🧹 Cleanup

### Quick Cleanup

```bash
./scripts/cleanup.sh
```

This removes:
- CloudFormation stack
- Lambda function
- Step Functions state machine
- IAM roles
- CloudWatch log groups (optional)

### Manual Cleanup

If script fails, remove resources manually:

```bash
# 1. Delete CloudFormation stack
aws cloudformation delete-stack --stack-name bioinfo-stepfunctions-demo

# 2. Wait for deletion
aws cloudformation wait stack-delete-complete --stack-name bioinfo-stepfunctions-demo

# 3. Delete log groups (if needed)
aws logs delete-log-group --log-group-name /aws/lambda/bioinfo-stepfunctions-demo-unreliable-api
aws logs delete-log-group --log-group-name /aws/vendedlogs/states/bioinfo-stepfunctions-demo-retry-demo
```

### Verification

Confirm all resources deleted:

```bash
# Should return error: Stack not found
aws cloudformation describe-stacks --stack-name bioinfo-stepfunctions-demo
```

---

## 🤝 Contributing

This is an educational project. Suggestions for improvements:

1. Fork the repository
2. Make your changes
3. Test thoroughly
4. Submit a pull request

Areas for contribution:
- Additional bioinformatics use cases
- Different retry strategies
- Integration with other AWS services
- Multi-region examples

---

## 📄 License

MIT License - Free to use for educational purposes

---

## 🎓 Next Steps

After completing this project:

1. **Read TUTORIAL.md** for deeper understanding
2. **Modify retry parameters** and observe behavior changes
3. **Integrate with real APIs** (NCBI, UniProt)
4. **Add S3 storage** for results
5. **Implement parallel processing** with Map states
6. **Build complete genomics pipeline**

---

## 📧 Support

Questions or issues?

1. Check `TUTORIAL.md` for detailed explanations
2. Review CloudWatch logs for execution details
3. Consult AWS documentation links above
4. Open an issue in the repository

---

## 🏆 Learning Checkpoints

Mark your progress:

- [ ] Successfully deployed infrastructure
- [ ] Executed state machine and observed retries
- [ ] Reviewed CloudWatch logs
- [ ] Understood exponential backoff calculations
- [ ] Modified retry parameters
- [ ] Read complete tutorial
- [ ] Cleaned up all resources
- [ ] Applied pattern to own project

**Congratulations on learning AWS Step Functions!** 🎉

---

**Last Updated:** 2026-01-05
**AWS SDK Version:** boto3 (Python 3.12)
**Region:** us-east-1 (free tier recommended)
