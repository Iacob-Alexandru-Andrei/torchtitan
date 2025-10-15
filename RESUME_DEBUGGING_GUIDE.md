# Resume Debugging Guide

## Problem
Training started at step 1 despite downloading checkpoint from step 10 and RESUME_FROM_RUN_STEP being set.

## Resume Flow in TorchTitan

### 1. Download Phase (Before training starts)
- **Location**: `experiments/fl/train.py` main function
- **Action**: `download_manager.download_if_needed()` is called
- **Result**: Downloads checkpoint from S3 to local `checkpoints/step-10/` directory
- **Verification**: Check logs for "✓ Checkpoint download complete for step X"

### 2. Training Initialization
- **Location**: `train.py` Trainer.__init__()
- **Initial State**: `self.step = 0` (always starts at 0)
- **Checkpoint Manager**: Created with `states={"train_state": self}`

### 3. Checkpoint Load Phase
- **Location**: `train.py` Trainer.train()
- **Action**: `self.checkpointer.load(step=job_config.checkpoint.load_step)`
- **Default load_step**: `-1` (means "load latest")
- **Steps**:
  1. `CheckpointManager.load()` is called with `step=-1`
  2. If checkpoint folder exists, calls `_find_load_step()` to find latest
  3. `_find_load_step()` scans for `step-*` directories, returns max (should be 10)
  4. Creates `checkpoint_id = "checkpoints/step-10"`
  5. Calls `_states_to_load(model_only=False)` which returns all states including "train_state"
  6. Calls `dcp.load(states, checkpoint_id=checkpoint_id)`
  7. DCP automatically calls `Trainer.load_state_dict()` which sets `self.step = 10`

### 4. Training Start
- **Location**: `train.py` Trainer.train()
- **Expected**: `self.step` should be 10 after load
- **Actual Behavior**: Logs say "Training starts at step {self.step + 1}"

## Debugging Steps

### Check 1: Verify Download
Look for logs:
```
✓ Checkpoint metadata file exists: checkpoints/step-10/.metadata
Downloaded checkpoint contains X total paths
✓ Checkpoint download complete for step 10
```

### Check 2: Verify Load Detection
Look for logs:
```
[RESUME DEBUG] Before checkpoint load: self.step = 0
[RESUME DEBUG] Checkpoint config: load_step=-1, folder=checkpoints
[RESUME DEBUG] States to load keys: ['train_state', 'optimizer', 'lr_scheduler', 'dataloader', ...]
[RESUME DEBUG] Model only: False
```

### Check 3: Verify State Restoration
Look for logs:
```
[RESUME DEBUG] Trainer.load_state_dict called with: {'step': 10, 'ntokens_seen': 320000}
[RESUME DEBUG] Before load: self.step = 0, self.ntokens_seen = 0
[RESUME DEBUG] After load: self.step = 10, self.ntokens_seen = 320000
```

### Check 4: Verify Final State
Look for logs:
```
[RESUME DEBUG] After checkpoint load: loaded=True, self.step = 10
Training starts at step 11
```

## Common Issues

### Issue 1: Checkpoint folder doesn't exist
**Symptom**: Download logs show success but `_find_load_step()` returns -1
**Cause**: Checkpoint downloaded to wrong location
**Fix**: Verify `self.checkpointer.folder` matches download destination

### Issue 2: .metadata file missing
**Symptom**: `_find_load_step()` doesn't find step-10
**Cause**: Incomplete checkpoint download
**Fix**: Check manifest and S3 upload logs

### Issue 3: train_state not in states_to_load
**Symptom**: load_state_dict never called
**Cause**: train_state excluded from loading
**Fix**: Check `exclude_from_loading` configuration

### Issue 4: DCP not calling load_state_dict
**Symptom**: Logs show states loaded but step still 0
**Cause**: Trainer not registered as Stateful
**Fix**: Verify Trainer inherits from `torch.distributed.checkpoint.stateful.Stateful`

## Verification Commands

### Check local checkpoint structure:
```bash
ls -la checkpoints/
ls -la checkpoints/step-10/
```

### Check checkpoint metadata:
```bash
cat checkpoints/step-10/.metadata
```

### Check latest marker:
```bash
cat checkpoints/latest
```

## Expected vs Actual Flow

### Expected:
1. Download step-10 → `checkpoints/step-10/` ✓
2. Load with step=-1 → finds step-10 ✓
3. DCP loads all states → train_state.step = 10 ✓
4. Training starts at step 11 ✓

### If Starting at Step 1:
Possible causes:
- Checkpoint not downloaded (download_if_needed skipped?)
- Checkpoint downloaded to wrong location
- load_state_dict not called
- load_state_dict called but state not restored
- Training loop reinitializes step

## Next Steps

1. Run training with RESUME_FROM_RUN_STEP set
2. Check all debug logs in sequence
3. Identify which step in the flow is failing
4. Fix based on the failure point identified
