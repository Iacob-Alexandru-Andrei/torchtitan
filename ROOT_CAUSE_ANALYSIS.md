# Root Cause Analysis: Resume Failure

## Problem
Training started at step 1 instead of resuming from step 10, despite:
- `RESUME_FROM_RUN_STEP="16M-baseline-20251011-122516/step-10"` being set
- S3 checkpoint existing and being accessible
- All resume code being in place

## Root Cause
**S3 checkpointing was completely disabled due to a boolean check bug!**

### The Bug
In `torchtitan/experiments/fl/train.py`, line 137-141:

```python
s3_checkpointing_active = (
    job_config.s3_checkpoint.enable
    and bool(job_config.s3_checkpoint.bucket)
    and bool(job_config.s3_checkpoint.prefix)  # ← BUG HERE!
)
```

### Configuration
From `mosaic_mup_16M.toml`:
```toml
[s3_checkpoint]
enable = true
bucket = "checkpoints"
prefix = ""  # Root of bucket
```

### The Problem
- `prefix = ""` (empty string) is a **valid configuration** meaning "use root of bucket"
- But `bool("")` evaluates to `False` in Python!
- So `s3_checkpointing_active = True and True and False = False`
- Result: **S3 manager never created, download never attempted**

## Evidence from Logs
```
[rank0]:[titan] 2025-10-11 13:23:24,943 - root - INFO - [RESUME DEBUG] Before checkpoint load: self.step = 0
[rank0]:[titan] 2025-10-11 13:23:24,943 - root - INFO - [RESUME DEBUG] Checkpoint config: load_step=-1, folder=checkpoints
[rank0]:[titan] 2025-10-11 13:23:24,943 - root - INFO - [RESUME DEBUG] After checkpoint load: loaded=False, self.step = 0
```

**Missing logs** (that should have appeared):
- "Checking for local checkpoints in: ..."
- "Resuming from run step: ..."
- "Downloading checkpoint step 10 from S3"

Why? Because `download_manager` was never created, so `download_if_needed()` was never called.

## The Fix
Change the condition to allow empty prefix:

```python
s3_checkpointing_active = (
    job_config.s3_checkpoint.enable
    and bool(job_config.s3_checkpoint.bucket)
    and job_config.s3_checkpoint.prefix is not None  # Empty string "" is valid!
)
```

Now:
- `prefix = ""` → `"" is not None` → `True` ✓
- `prefix = "torchtitan"` → `"torchtitan" is not None` → `True` ✓
- `prefix = None` → `None is not None` → `False` ✓

## Impact
This bug completely broke S3 checkpointing for any configuration using an empty prefix (root of bucket).
- No downloads from S3
- No uploads to S3
- Silent failure (no error message)

## Lesson Learned
**Validate configuration with realistic test cases!**

Empty strings are often valid values (paths, prefixes, etc.) and should not be checked with `bool()`.

Use instead:
- `is not None` for "must be provided"
- `!= ""` for "must not be empty"
- `bool()` only for truly binary flags

## Next Steps
1. Apply the fix
2. Run training again
3. Verify S3 download logs appear
4. Verify training resumes from step 10
5. Add configuration validation to catch this earlier
