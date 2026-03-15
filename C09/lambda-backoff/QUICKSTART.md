# Quick Start Guide

## 5-Minute Setup

### Step 1: Prerequisites
```bash
# Check AWS CLI
aws --version

# Check jq
jq --version

# Configure AWS (if not already done)
aws configure
```

### Step 2: Deploy
```bash
chmod +x scripts/*.sh
./scripts/deploy.sh
```
**Wait:** 2-3 minutes

### Step 3: Execute
```bash
./scripts/execute.sh
```
**Watch:** Retry attempts in real-time

### Step 4: View Logs
```bash
aws logs tail /aws/lambda/bioinfo-stepfunctions-demo-unreliable-api --follow
```

### Step 5: Clean Up
```bash
./scripts/cleanup.sh
```

---

## Common Commands

### Deploy
```bash
./scripts/deploy.sh [stack-name]
```

### Execute
```bash
./scripts/execute.sh [stack-name] [sample-id]
```

### View Logs
```bash
aws logs tail /aws/lambda/<function-name> --follow
```

### Cleanup
```bash
./scripts/cleanup.sh [stack-name]
```

---

## What You'll Learn

✓ Exponential backoff retry logic
✓ AWS Step Functions orchestration
✓ Lambda error handling
✓ CloudWatch monitoring
✓ Infrastructure as Code

---

## Next Steps

1. Read `README.md` for complete overview
2. Study `TUTORIAL.md` for deep dive
3. Check `examples/sample-output.txt` for expected results
4. Modify retry parameters and observe changes

---

## Cost

**$0.00** - Everything is free tier compatible!

---

## Support

Check `README.md` troubleshooting section or review CloudWatch logs.
