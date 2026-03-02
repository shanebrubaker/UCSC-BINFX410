# Nextflow Tower / Seqera Platform — Setup Guide

This guide walks you through connecting the Sequence QC Pipeline to
**Seqera Platform** (formerly Nextflow Tower) so you can monitor pipeline
runs in real time from a browser.

---

## What is Seqera Platform / Tower?

Seqera Platform (https://cloud.seqera.io) is the official cloud monitoring
and management service for Nextflow pipelines. When a pipeline runs with
Tower enabled you get:

- **Live progress** — task-by-task status updated in real time
- **Resource metrics** — CPU, memory, and I/O per process
- **Execution timeline** — interactive Gantt chart
- **Log streaming** — tail process logs without SSH
- **Run history** — compare past executions side-by-side

---

## Quick Start (5 steps)

### Step 1 — Create a free Seqera account

Go to https://cloud.seqera.io and sign up with GitHub, Google, or email.
The free tier is sufficient for this exercise.

### Step 2 — Generate an access token

1. Log in to https://cloud.seqera.io
2. Click your avatar → **Your tokens** (or go to https://cloud.seqera.io/tokens)
3. Click **Add token**, give it a name (e.g. `binfx410-local`), and copy the token.

> **Security note**: treat your token like a password — do not commit it to
> git or share it publicly.

### Step 3 — Export the token in your shell

```bash
export TOWER_ACCESS_TOKEN="eyJ..."   # paste your token here
```

Add that line to `~/.zshrc` (or `~/.bashrc`) to make it permanent:

```bash
echo 'export TOWER_ACCESS_TOKEN="eyJ..."' >> ~/.zshrc
source ~/.zshrc
```

### Step 4 — Test the connection

Run the included test script:

```bash
./test_tower.sh
```

The script will:
1. Verify `curl`, `nextflow`, and `python3` are available
2. Confirm your token is set
3. Call the Tower API and print your username
4. List your accessible workspaces
5. Generate small test data
6. Run the pipeline with `-with-tower` and print a link to your run

### Step 5 — Watch your run in the browser

Open the link printed at the end of the test script, or navigate to
https://cloud.seqera.io/user/runs directly.

---

## Running the Pipeline with Tower Monitoring

### Option A — Flag on the command line (recommended for learning)

```bash
export TOWER_ACCESS_TOKEN="eyJ..."
nextflow run main.nf -with-tower
```

### Option B — Tower profile (bakes in Docker + Tower together)

```bash
export TOWER_ACCESS_TOKEN="eyJ..."
nextflow run main.nf -profile tower
```

### Option C — Enable Tower for every run automatically

Edit `nextflow.config` and change `tower.enabled` to `true`:

```groovy
tower {
    enabled     = true   // ← change false → true
    accessToken = System.getenv('TOWER_ACCESS_TOKEN') ?: ''
}
```

Then every `nextflow run` will automatically report to Tower as long as
`TOWER_ACCESS_TOKEN` is set.

---

## Optional — Target a Specific Workspace

If you belong to an organisation on Seqera Platform and want runs to appear
in a shared workspace instead of your personal namespace:

```bash
# Find your numeric workspace ID from the test script output or from the URL
# when you visit the workspace: cloud.seqera.io/orgs/<org>/workspaces/<ws>
export TOWER_WORKSPACE_ID="12345678"
nextflow run main.nf -with-tower -tower-workspace-id "$TOWER_WORKSPACE_ID"
```

Or add it permanently to `nextflow.config`:

```groovy
tower {
    enabled     = true
    accessToken = System.getenv('TOWER_ACCESS_TOKEN') ?: ''
    workspaceId = System.getenv('TOWER_WORKSPACE_ID') ?: ''
}
```

---

## Self-Hosted Seqera Platform

If your institution runs its own Seqera Platform instance, point the client
at your server's API instead of the public cloud:

```bash
export TOWER_API_ENDPOINT="https://seqera.your-org.edu/api"
export TOWER_ACCESS_TOKEN="eyJ..."
nextflow run main.nf -with-tower
```

The `nextflow.config` already reads `TOWER_API_ENDPOINT` from the
environment, so no config changes are needed.

---

## File Reference

| File | Purpose |
|------|---------|
| `nextflow.config` | `tower {}` block + `tower` profile |
| `main.nf` | Tower-aware startup log and run-URL in `onComplete` |
| `test_tower.sh` | End-to-end connection test + pipeline run |
| `run_test.sh` | Original local test (no Tower) |
| `TOWER_SETUP.md` | This file |

---

## Troubleshooting

### "TOWER_ACCESS_TOKEN is not set"

```bash
export TOWER_ACCESS_TOKEN="your_token_here"
```

### "HTTP 401 — Invalid or expired access token"

The token may have been revoked or has expired.  Generate a new one at
https://cloud.seqera.io/tokens.

### "Could not reach https://api.cloud.seqera.io"

- Check that you have internet access
- Check that a firewall or VPN is not blocking outbound HTTPS
- For a self-hosted instance, confirm `TOWER_API_ENDPOINT` is correct

### Run does not appear in the Tower dashboard

1. Confirm the pipeline actually started (check the Nextflow log for
   `[TOWER]` lines at startup).
2. Verify `TOWER_ACCESS_TOKEN` is set in the same shell session.
3. Try passing the token explicitly:

```bash
nextflow run main.nf -with-tower -tower-token "$TOWER_ACCESS_TOKEN"
```

### Pipeline runs fine but Tower shows no metrics

Tower collects resource metrics only when Nextflow can report them.  If you
are running without a container executor (Docker/Singularity), resource data
(CPU %, memory) may be absent.  Use `-profile tower` to enable Docker
automatically.

---

## Further Reading

- Seqera Platform docs: https://docs.seqera.io/platform
- Nextflow `-with-tower` flag: https://www.nextflow.io/docs/latest/monitoring.html
- Token management: https://docs.seqera.io/platform/23.3/api/overview/#authentication
