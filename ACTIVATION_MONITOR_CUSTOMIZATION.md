# Activation Monitor Customization

## Overview

The `ActivationMonitor` class in `torchtitan/experiments/fl/metrics.py` now supports customizable metrics collection. By default, only essential metrics are collected to reduce overhead.

## Default Enabled Metrics

By default, only the following 8 metrics are collected:

1. `activations/average/max/full_model_input`
2. `activations/average/max/full_model_output`
3. `activations/average/median/full_model_input`
4. `activations/average/median/full_model_output`
5. `activations/l2_norm/full_model_input`
6. `activations/l2_norm/full_model_output`
7. `activations/max/full_model_input`
8. `activations/max/full_model_output`

All other activation metrics (e.g., skewness, kurtosis, min values) are **disabled by default**.

## Configuration

To customize which metrics are collected, add the following to your job configuration:

```toml
# Enable activation monitoring
activation_monitor_enabled = true
activation_monitor_interval = 25

# Customize which metrics to collect (optional)
# If not specified, uses the default 8 metrics listed above
activation_monitor_enabled_metrics = [
    "activations/average/max/full_model_input",
    "activations/average/max/full_model_output",
    "activations/average/median/full_model_input",
    "activations/average/median/full_model_output",
    "activations/l2_norm/full_model_input",
    "activations/l2_norm/full_model_output",
    "activations/max/full_model_input",
    "activations/max/full_model_output",
    # Add any additional metrics you want to enable:
    # "activations/average/min/full_model_input",
    # "activations/skewness/max/full_model_input",
    # "activations/kurtosis/median/full_model_output",
]
```

## Available Metrics

### Base Metrics
- `activations/l2_norm/full_model_input`
- `activations/l2_norm/full_model_output`
- `activations/max/full_model_input`
- `activations/max/full_model_output`

### Average Statistics
- `activations/average/max/full_model_input`
- `activations/average/max/full_model_output`
- `activations/average/min/full_model_input`
- `activations/average/min/full_model_output`
- `activations/average/median/full_model_input`
- `activations/average/median/full_model_output`

### Skewness Statistics
- `activations/skewness/max/full_model_input`
- `activations/skewness/max/full_model_output`
- `activations/skewness/min/full_model_input`
- `activations/skewness/min/full_model_output`
- `activations/skewness/median/full_model_input`
- `activations/skewness/median/full_model_output`

### Kurtosis Statistics
- `activations/kurtosis/max/full_model_input`
- `activations/kurtosis/max/full_model_output`
- `activations/kurtosis/min/full_model_input`
- `activations/kurtosis/min/full_model_output`
- `activations/kurtosis/median/full_model_input`
- `activations/kurtosis/median/full_model_output`

## Implementation Details

### Code Changes

1. **`ActivationMonitor.__init__`**: Added `enabled_metrics` parameter
   - Accepts an optional `set[str]` of metric names to enable
   - If `None`, uses the default set of 8 essential metrics

2. **`ActivationMonitor._is_metric_enabled`**: New method
   - Checks if a specific metric is enabled for collection
   - Returns `True` if the metric is in the `enabled_metrics` set

3. **`ActivationMonitor._add_metrics`**: Modified to check enabled status
   - Before collecting any metric, checks if it's enabled
   - Skips computation for disabled metrics to save performance

4. **`ActivationMonitor._prepare_local_metrics`**: Modified to respect enabled metrics
   - Only prepares metrics that are enabled
   - Ensures disabled metrics are not included in final output

5. **`FLMetricsProcessor.__init__`**: Updated to read config
   - Reads `activation_monitor_enabled_metrics` from job config
   - Passes the enabled metrics set to `ActivationMonitor`

### Performance Benefits

By collecting only essential metrics by default:
- **Reduced computation**: Skewness and kurtosis calculations are skipped when not needed
- **Lower memory usage**: Fewer metric lists are maintained
- **Faster logging**: Less data to reduce and log across ranks
- **Configurable overhead**: Users can enable more metrics only when needed for detailed analysis

## Example Usage

### Minimal Configuration (uses defaults)
```toml
activation_monitor_enabled = true
```

### Custom Configuration
```toml
activation_monitor_enabled = true
activation_monitor_interval = 50

# Enable all statistics for detailed analysis
activation_monitor_enabled_metrics = [
    "activations/l2_norm/full_model_input",
    "activations/l2_norm/full_model_output",
    "activations/max/full_model_input",
    "activations/max/full_model_output",
    "activations/average/max/full_model_input",
    "activations/average/max/full_model_output",
    "activations/average/min/full_model_input",
    "activations/average/min/full_model_output",
    "activations/average/median/full_model_input",
    "activations/average/median/full_model_output",
    "activations/skewness/max/full_model_input",
    "activations/skewness/median/full_model_output",
    "activations/kurtosis/max/full_model_input",
    "activations/kurtosis/median/full_model_output",
]
```

### Disable All Activation Monitoring
```toml
activation_monitor_enabled = false
```
