---
name: "PACE Phoenix"
source: "https://docs.pace.gatech.edu"
category: "infrastructure"
relevance: "HPC cluster for training and inference workloads"
status: "active"
---

# PACE Phoenix Slurm Reference

Quick reference for using Slurm on the Georgia Tech PACE Phoenix cluster. Covers job submission, GPU resources, and common patterns for ML workloads.

Last updated: January 2026. For the most current information, consult the [official PACE documentation](https://docs.pace.gatech.edu).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Common Commands](#common-commands)
3. [Decision Guides](#decision-guides)
4. [Informational Commands](#informational-commands)
5. [Job Accounting](#job-accounting)
6. [QoS Policies](#qos-policies)
7. [Interactive Jobs](#interactive-jobs)
8. [Batch Jobs](#batch-jobs)
9. [Array Jobs](#array-jobs)
10. [GPU Jobs](#gpu-jobs)
11. [Local Disk Jobs](#local-disk-jobs)
12. [Best Practices](#best-practices)
13. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Simple CPU Job (Free)

```bash
salloc -A gts-gburdell3 -q embers -N1 --ntasks-per-node=4 -t 1:00:00
```

### GPU Job (Paid)

```bash
salloc -A gts-gburdell3 -q inferno -N1 --gres=gpu:V100:1 --mem-per-gpu=12G -t 2:00:00
```

### Batch Script Template

```bash
#!/bin/bash
#SBATCH -J my_job
#SBATCH -A gts-gburdell3
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2G
#SBATCH -t 1:00:00
#SBATCH -q embers
#SBATCH -o output-%j.log

srun python my_script.py
```

---

## Common Commands

| Task | Command |
|------|---------|
| Check your jobs | `squeue -u $USER` |
| Check account balance | `pace-quota` |
| Check queue status | `pace-check-queue inferno` |
| Submit batch job | `sbatch script.sbatch` |
| Cancel job | `scancel <jobid>` |
| Job details | `scontrol show job <jobid>` |
| Past job info | `sacct -j <jobid>` |
| Job summary | `pace-job-summary <jobid>` |
| Resource efficiency | `seff <jobid>` |

---

## Decision Guides

### Which QoS Should I Use?

| Scenario | QoS | Why |
|----------|-----|-----|
| Development/testing | **embers** | Free, 8hr limit |
| Production work | **inferno** | No preemption, 3-21 days |
| Quick debugging | **embers** | Free tier |
| Critical deadline | **inferno** | Higher priority |

### Which GPU Should I Request?

| Workload | GPU | Why |
|----------|-----|-----|
| Small models, prototyping | RTX 6000 (24GB) | Widely available |
| Medium models | V100-32GB or A100-40GB | Good balance |
| Large models | A100-80GB or H100 (80GB) | High VRAM |
| Very large models | H200 (142GB) | Maximum VRAM |

### Which Partition Should I Use?

| Workload | Partition | Constraint |
|----------|-----------|------------|
| General CPU | `cpu-small` or `cpu-gnr` | None |
| High-memory CPU | `cpu-small` | None, request more mem |
| V100 GPU | `gpu-v100` | `--gres=gpu:V100:1` |
| RTX 6000 GPU | `gpu-rtx6000` | `--gres=gpu:RTX_6000:1` |
| A100 GPU | `gpu-a100` | `--gres=gpu:A100:1` |
| Large local disk | Any | `-C localSAS` |

---

## Informational Commands

### squeue

Check job status for pending (PD) and running (R) jobs.

**Common options:**
- `-j <job number>` — Show information about specific jobs (comma-separated for multiple)
- `-u <username>` — Show jobs belonging to a specific user
- `-A <charge account>` — See jobs belonging to a specific charge account
- `-p <partition>` — See jobs submitted to a specific partition
- `-q <QOS>` — See jobs submitted to a specific QoS

**Documentation:** `man squeue` or [squeue docs](https://slurm.schedmd.com/squeue.html)

### sacct

After a job has completed, use `sacct` to find information about it.

**Common options:**
- `-j <job number>` — Find information about specific jobs
- `-u <username>` — See all jobs belonging to a specific user
- `-A <charge account>` — See jobs belonging to a specific charge account
- `-X` — Show information only about the allocation, not steps inside it
- `-S <time>` — List jobs only after a specified time (formats: `YYYY-MM-DD[HH:MM[:SS]]`)
- `-o <fields>` — Specify which columns of data should appear in output

**Documentation:** `man sacct` or [sacct docs](https://slurm.schedmd.com/sacct.html)

### scancel

Cancel a job:

```bash
scancel <job number>
```

### pace-check-queue

Provides an overview of current utilization of each partition's nodes.

```bash
pace-check-queue inferno      # For QoS
pace-check-queue cpu-small    # For partition
```

Options:
- `-s` — See all features of each node in the partition
- `-c` — Color-code the "Accepting Jobs?" column

### pace-job-summary

High-level overview about a completed job:

```bash
pace-job-summary <JobID>
```

### pace-quota

Find your Phoenix-Slurm charge accounts and storage allocations:

```bash
pace-quota
```

Shows:
- **Balance:** Current total based on completed transactions
- **Reserved:** Sum of liens based on running jobs
- **Available:** Total funds available for new job submissions

Account naming convention: Most accounts are of the form `gts-<PI username>[-<descriptor>]`

---

## Job Accounting

### How Charging Works

Phoenix's accounting system is based on the most significant processing unit:

- **CPU nodes:** Charge rates based on CPU-hours (total procs x walltime)
- **GPU nodes (V100, RTX6000, A100, H100, L40S):** Charge rates based on GPU-hours (total GPUs x walltime)

### Specifying Charge Account

The account **must be specified** using the `-A` flag:
- Command line: `salloc -A gts-gburdell3 ...`
- Batch file: `#SBATCH -A gts-gburdell3`

The scheduler will:
1. Verify the account has sufficient funds for the full job length
2. Place a lien on the account when the job starts
3. Release excess funds if the job finishes early

### Account Types

| Account Syntax | Description |
|----------------|-------------|
| `gts-<PI UID>` | Institute-sponsored account with 10k CPU hours (credits reset monthly) |
| `gts-<PI UID>-CODA20` | 2020 hardware refresh account |
| `gts-<PI UID>-<group>` | PI-specific child account for shared account |
| `gts-<PI UID>-<custom>` | Postpaid or prepaid billing model account |

### Resource Rates

Rates for each node class are at the [PACE resources page](https://docs.pace.gatech.edu/phoenix_cluster/gettingstarted_phnx/#compute-resources).

Node classes include: CPU-192GB, CPU-768GB-SAS, GPU-192GB-V100, GPU-384GB-RTX6000, GPU-A100, GPU-H100, GPU-H200, GPU-L40S.

---

## QoS Policies

Two QoS levels on Phoenix: **inferno** and **embers**. Both provide access to the same resource pool but with different job policies.

### Inferno (Paid, High Priority)

Main production QoS — jobs consume account credits but get priority treatment.

| Property | Value |
|----------|-------|
| Base priority | 250,000 |
| Max jobs per user | 500 |
| Max walltime (CPU) | 21 days |
| Max walltime (GPU) | 3 days |
| Preemption | No |

**Note:** The scheduler rejects a job if `nodes x cores x walltime > 264,960 processor-hours`.

```bash
# Explicitly specify (or omit — inferno is default)
salloc -q inferno ...
#SBATCH -q inferno
```

### Embers (Free, Backfill)

Free backfill QoS — opportunistic scheduling, no credits consumed.

| Property | Value |
|----------|-------|
| Base priority | 0 (lowest) |
| Max jobs per user | 50 |
| Max eligible jobs per user | 1 |
| Max walltime | 8 hours |
| Preemption | After 1 hour |

```bash
salloc -q embers ...
#SBATCH -q embers
```

### Concurrent Job Limits

| Constraint | Limit | Applies To |
|------------|-------|------------|
| Per-charge-account processors | 6000 | inferno only |
| Per-user GPUs | 32 | inferno only |
| Per-charge-account CPU-time | 300,000 CPU-hours | Both |

### Quick Comparison

| Feature | Inferno | Embers |
|---------|---------|--------|
| **Cost** | Consumes credits | FREE |
| **Priority** | 250,000 (high) | 0 (low) |
| **Max Walltime** | 3-21 days | 8 hours |
| **Preemption** | No | Yes (after 1 hour) |
| **Max Jobs** | 500 | 50 |
| **Best For** | Production | Development/Testing |

---

## Interactive Jobs

Interactive jobs reserve resources on compute nodes for interactive use.

### Using salloc

**Required parameters:**
- `--account` or `-A` — Charge account
- `--qos` or `-q` — Quality of Service (inferno or embers)

**Common optional parameters:**
- `--nodes` or `-N` — Number of nodes
- `--ntasks-per-node` — Cores per node
- `-n` — Total cores
- `--time` or `-t` — Wall time (format: `D-HH:MM:SS`)
- `--mem-per-cpu` — Memory per core
- `--partition` or `-p` — Partition
- `--x11` — Enable X forwarding for GUI applications

**Documentation:** `man salloc` or [salloc docs](https://slurm.schedmd.com/salloc.html)

### Basic Example

Request 1 node with 4 cores for 1 hour:

```bash
salloc -A gts-gburdell3 -q inferno -N1 --ntasks-per-node=4 -t1:00:00
```

Once allocated, you're on the compute node. Use `srun` to run commands:

```bash
srun hostname
```

### CPU Partition Example

```bash
salloc -A gts-gburdell3 -q embers -p cpu-gnr -N1 --ntasks-per-node=4 -t1:00:00
```

### GPU Interactive Example

Request 1 node with a V100 GPU:

```bash
salloc -A gts-gburdell3 -N1 --mem-per-gpu=12G -q inferno -t0:15:00 --gres=gpu:V100:1
```

**Note:** No need to specify `--ntasks-per-node` for GPUs — cores are assigned automatically:
- 6 cores per RTX6000
- 12 cores per V100
- 8 cores per A100, H100, or H200
- 4 cores per L40S

### Exiting

Type `exit` or press `Ctrl-D` to release the allocation.

---

## Batch Jobs

Batch jobs are scripts submitted to Slurm for automated execution.

### Common Directives

```bash
#!/bin/bash
#SBATCH -J <job name>                    # Job name
#SBATCH --account=<account>              # Charge account
#SBATCH -N <nodes>                       # Number of nodes
#SBATCH --ntasks-per-node=<cores>        # Cores per node
#SBATCH --mem-per-cpu=<size>             # Memory per core (e.g., 1G)
#SBATCH --mem-per-gpu=<size>             # Memory per GPU (for GPU jobs)
#SBATCH -t <time>                        # Walltime (D-HH:MM:SS or minutes)
#SBATCH -q <qos>                         # QoS (inferno or embers)
#SBATCH -p <partition>                   # Partition
#SBATCH -o <filename>                    # Output file
#SBATCH --mail-type=BEGIN,END,FAIL       # Email notifications
#SBATCH --mail-user=<email>              # Email address
```

### Resource Requests

**CPU cores:**
```bash
#SBATCH -n 4                 # 4 total cores
# OR
#SBATCH -N 2                 # 2 nodes
#SBATCH --ntasks-per-node=4  # 4 cores per node (8 total)
```

**Memory:**
```bash
#SBATCH --mem-per-cpu=1G     # For CPU jobs
#SBATCH --mem-per-gpu=12G    # For GPU jobs
#SBATCH --mem=0              # All memory on node
```

**Walltime:**
```bash
#SBATCH -t 1:30:00           # 1 hour 30 minutes
#SBATCH -t 2-00:00:00        # 2 days
#SBATCH -t 15                # 15 minutes (integer = minutes)
```

### Submit and Monitor

```bash
sbatch script.sbatch         # Submit
squeue -u $USER              # Check status
scancel <jobid>              # Cancel
```

### Default Values

- **Cores:** 1 core (if not specified)
- **Memory:** 1GB per core (if not specified)
- **Walltime:** 1 hour (if not specified)
- **QoS:** inferno (if not specified)

### Using srun

Prefix computationally-intensive commands with `srun` in batch scripts:

```bash
srun python script.py     # Runs on allocated resources
```

---

## Array Jobs

Submit multiple identical jobs without external scripting.

**Maximum: 500 jobs (queued + running) per user.**

### Syntax

```bash
#SBATCH --array=1-100        # Range from 1 to 100
#SBATCH --array=4,8,15,42   # Specific tasks only
#SBATCH --array=1-200%5     # 200 tasks, max 5 running at once
```

The `%N` suffix limits active tasks to N at any time.

### Using the Task ID

Each task gets `$SLURM_ARRAY_TASK_ID`:

```bash
# Numbered files
srun process_file.py file${SLURM_ARRAY_TASK_ID}.txt

# From a list file (one item per line)
SAMPLE_LIST=($(<input.list))
SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}
srun process.py $SAMPLE
```

### Output Files

Use `%A` (master job ID) and `%a` (task ID):

```bash
#SBATCH --output=Array_test.%A_%a.out
```

**Critical:** Always use both `%A` and `%a` in log file names. Using only `%A` causes all tasks to write to a single file, drastically reducing performance.

### Handling Many Short Tasks

Instead of 500 array tasks, batch short tasks with loops:

```bash
#!/bin/bash
#SBATCH --array=1-5              # Only 5 tasks
#SBATCH -A gts-gburdell3
#SBATCH -t 00:10:00

PER_TASK=100
START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))
END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))

for (( run=$START_NUM; run<=END_NUM; run++ )); do
    srun python test.py --run=$run
done
```

### Modifying Running Arrays

```bash
scontrol update ArrayTaskThrottle=50 JobId=123456
scontrol update ArrayTaskThrottle=0 JobId=123456    # Remove limit
```

### Deleting Array Jobs

```bash
scancel 123456       # Delete all tasks
scancel 123456_1     # Delete single task
```

### Common Patterns

**Parameter sweep:**
```bash
#SBATCH --array=1-10
PARAM=$(echo "scale=2; $SLURM_ARRAY_TASK_ID * 0.1" | bc)
srun simulation --parameter=$PARAM
```

**Process files in directory:**
```bash
#SBATCH --array=1-100
FILES=($(ls data/*.txt))
FILE=${FILES[$SLURM_ARRAY_TASK_ID-1]}
srun process.sh $FILE
```

---

## GPU Jobs

### Available GPU Types

| GPU Model | Constraint Flag | GRES Flag | VRAM | Cores/GPU |
|-----------|----------------|-----------|------|-----------|
| **V100** (16GB) | `-C V100-16GB` | `--gres=gpu:V100:N` | 16GB | 12 |
| **V100** (32GB) | `-C V100-32GB` | `--gres=gpu:V100:N` | 32GB | 12 |
| **RTX 6000** | `-C RTX6000` | `--gres=gpu:RTX_6000:N` | 24GB | 6 |
| **A100** (40GB) | `-C A100-40GB` | `--gres=gpu:A100:N` | 40GB | 8 (up to 32) |
| **A100** (80GB) | `-C A100-80GB` | `--gres=gpu:A100:N` | 80GB | 8 (up to 32) |
| **H100** | `-C H100` | `--gres=gpu:H100:N` | 80GB | 8 |
| **H200** | `-C H200` | `--gres=gpu:H200:N` | 142GB | 8 |
| **L40S** | `-C L40S` | `--gres=gpu:L40S:N` | 48GB | 4 |
| **RTX Pro Blackwell** | `-C RTX-Pro-Blackwell` | `--gres=gpu:rtx_pro_6000_blackwell:N` | - | - |

**Note:** Some GPU nodes (8 of 10 L40S, 1 A100, 1 H100) are only available on `embers` QoS.

### Requesting GPUs

**Method 1: GRES (GPUs per node)**
```bash
--gres=gpu:V100:2          # 2 V100 GPUs per node
--gres=gpu:RTX_6000:1      # 1 RTX 6000 per node
--gres=gpu:A100:4          # 4 A100 GPUs per node
--gres=gpu:H100:1          # 1 H100 per node
--gres=gpu:L40S:2          # 2 L40S per node
```

**Method 2: Total GPUs with constraint**
```bash
-G 2 -C V100-32GB          # 2 V100 32GB GPUs total
-G 1 -C RTX6000            # 1 RTX 6000
-G 4 -C gpu-a100           # 4 A100 GPUs (any memory)
-G 1 -C HX00               # First available H100 or H200
```

### CPU Cores per GPU (Automatic)

| GPU Type | Default Cores | Max Cores |
|----------|---------------|-----------|
| RTX 6000 | 6 | 6 (fixed) |
| V100 | 12 | 12 (fixed) |
| A100 | 8 | 32 (configurable) |
| H100 | 8 | 8 (fixed) |
| H200 | 8 | 8 (fixed) |
| L40S | 4 | 4 (fixed) |

For A100 only, you can request up to 32 cores: `--gres=gpu:A100:1 --ntasks-per-node=32`

### Memory

Use `--mem-per-gpu` for GPU jobs:

```bash
--mem-per-gpu=12G
--mem-per-gpu=48G
```

### Interactive GPU Example

```bash
salloc -A gts-gburdell3 -N1 --mem-per-gpu=12G -q inferno -t 0:15:00 --gres=gpu:V100:1
```

### Batch GPU Example

```bash
#!/bin/bash
#SBATCH -J GPU_example
#SBATCH -A gts-gburdell3
#SBATCH -N1
#SBATCH --gres=gpu:V100:1
#SBATCH --mem-per-gpu=12GB
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -o Report-%j.out

module load cuda
srun ./my_gpu_program
```

### Flexible GPU Requests

Accept the first available GPU when resources are scarce:

```bash
salloc -A gts-gburdell3 -G 1 -C 'A100|V100|RTX6000'
```

Check actual GPU assigned:

```bash
scontrol show job $SLURM_JOB_ID | grep Partition
```

### Common GPU Request Examples

```bash
# RTX 6000 for 2 hours
salloc -A gts-gburdell3 -N1 --gres=gpu:RTX_6000:1 --mem-per-gpu=20G -q inferno -t 2:00:00

# A100 with 32 cores
salloc -A gts-gburdell3 -N1 --gres=gpu:A100:1 --ntasks-per-node=32 --mem-per-gpu=48G -q inferno -t 4:00:00

# Free L40S on embers
salloc -A gts-gburdell3 -N1 --gres=gpu:L40S:1 --mem-per-gpu=24G -q embers -t 4:00:00
```

### Constraint Options Summary

**Generic (any memory):**
- `-C gpu-v100` — V100 (16GB or 32GB)
- `-C gpu-rtx6000` — RTX 6000
- `-C gpu-a100` — A100 (40GB or 80GB)
- `-C gpu-h100` — H100
- `-C gpu-h200` — H200
- `-C HX00` — First available H100 or H200
- `-C gpu-l40s` — L40S

**Specific memory:**
- `-C V100-16GB`, `-C V100-32GB`
- `-C A100-40GB`, `-C A100-80GB`

### Checking GPU Availability

```bash
pace-check-queue gpu-v100
pace-check-queue gpu-a100
pace-check-queue gpu-h100
```

---

## Local Disk Jobs

Every Phoenix node has local disk storage for temporary use, automatically cleared upon job completion.

### Why Use Local Disk?

Local disk provides faster I/O than network storage (home, scratch) for:
- Temporary working files
- Intermediate computation results
- High-frequency read/write operations
- Python virtual environments (significantly faster imports)

### Storage Types

| Node Type | Storage Type | Capacity |
|-----------|--------------|----------|
| Standard nodes | NVMe | 1 TB |
| localSAS nodes | SAS | 8 TB |

### Using $TMPDIR

Slurm automatically creates a temporary directory for each job. The directory is **automatically deleted** when the job ends.

```bash
# In batch script or interactive session
cd ${TMPDIR}

# Copy data to local disk
cp ~/scratch/data/input.txt ${TMPDIR}/

# Process on local disk (fast!)
srun ./process_data input.txt > results.txt

# Copy results back before job ends
cp results.txt ~/scratch/output/
```

### Requesting Local Disk Space

For partial node requests, guarantee availability:

```bash
#SBATCH --tmp=100G          # Request 100GB local disk
#SBATCH --tmp=50000         # 50,000 MB (default unit is MB)
```

### Requesting SAS Storage Nodes

For jobs needing 8 TB local disk:

```bash
#SBATCH -C localSAS
#SBATCH --tmp=5T
```

### Performance Comparison

| Storage Type | Typical Speed | Use Case |
|--------------|---------------|----------|
| **Local NVMe** | Very fast | Temporary, high I/O |
| **Local SAS** | Fast | Large temporary files |
| **Home** | Moderate | Permanent, small files |
| **Scratch** | Moderate-Fast | Large permanent files |

### Important Notes

- Data is **deleted** when job ends — always copy results out
- Not for long-term storage — use home or scratch for permanent data
- Private to your job — other users can't access your `$TMPDIR`

---

## Best Practices

### Resource Requests

- **Request only what you need** — don't request entire nodes if you only need a few cores
- **Estimate walltime accurately** — jobs finishing early release resources; too short kills the job
- **Use appropriate memory** — monitor actual usage with `seff <jobid>` after completion
- **Choose right QoS** — use embers for testing, inferno for production

### Job Submission

- **Test with small jobs first** — validate your script works before large-scale runs
- **Use array jobs wisely** — batch short tasks together, throttle large arrays
- **Monitor your jobs** — check status with `squeue`, review output files
- **Clean up after yourself** — cancel unnecessary jobs, check for zombie processes

### Performance

- **Use `srun`** — always prefix compute commands with `srun` in batch scripts
- **Use local disk** — for I/O intensive work, use `$TMPDIR`
- **Load modules** — ensure all dependencies are loaded before execution

---

## Troubleshooting

| Problem | Check | Solution |
|---------|-------|----------|
| Job pending forever | Queue status | Try different partition/QoS, or use flexible GPU constraints |
| Job fails immediately | Output file | Check error messages |
| Out of memory | Resource usage | Increase `--mem-per-cpu` or `--mem-per-gpu` |
| Out of time | Walltime | Increase `-t` value |
| No credits | Account balance | Run `pace-quota`, contact PI |
| Wrong GPU assigned | Constraint flags | Be specific (e.g., `-C A100-80GB` vs `-C gpu-a100`) |

### Useful Debug Commands

```bash
# Why is my job pending?
scontrol show job <jobid>

# What resources did I use?
pace-job-summary <jobid>
seff <jobid>

# Check queue availability
pace-check-queue inferno
pace-check-queue embers
```

### Contact Support

- **PACE Documentation:** https://docs.pace.gatech.edu
- **Support Email:** pace-support@oit.gatech.edu

---

## Notes

- All examples use `gts-gburdell3` as the account — **replace with your actual account**
- Use `pace-quota` to find your account names
- Always test new scripts with short walltimes first
- GPU jobs are charged by GPU-hours, not CPU-hours
