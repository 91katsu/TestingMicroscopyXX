# Monte Carlo Uncertainty Estimation

Monte Carlo (MC) inference runs the model multiple times under stochastic conditions and uses the disagreement between passes to produce a per-voxel uncertainty map (`xystd`). High-std regions correspond to uncertain boundaries of cellular structures.

## How it works

Two independent sources of variance contribute to the uncertainty estimate:

1. **MC Dropout**: The model is loaded directly from checkpoint without switching to eval mode (`model.eval()` is never called), so dropout layers and batchnorm batch-statistics remain active during inference. Each forward pass uses a different random dropout mask, producing different outputs for the same input.
2. **Test-time augmentation (TTA)**: Each entry in `tta_method` applies a spatial transform before the model and inverts it after, exposing prediction inconsistencies under symmetry. Available options: `null` (identity), `"transpose"` (swap X / Y), `"flipX"`, `"flipY"`, `"flipZ"`.

Total forward passes per patch:

```
total passes = len(tta_method) × num_mc
```

For example, `tta_method: [null, "transpose"]` and `num_mc: 3` runs 6 passes (each TTA setting repeated 3 times under different dropout masks).

Each pass is binarized at `mc_threshold`, and the per-voxel standard deviation across all passes is reported as `xystd`.

## How to use

### Step 1: Determine `mc_threshold`

Run normal inference and inspect `xy` to choose a threshold that separates structure from background.

```yaml
# Step 1 config — probe xy values to choose threshold
output:
  save: ["xy"]
  output_datatype: "float32"   # required: read raw values directly
model:
  num_mc: 1                    # MC not needed at this stage
```

```bash
python test.py --env MyEnv --scale 8x --override MyExperiment
```

Note that `mc_threshold` operates in the same float32 range as `xy`. We recommend setting `output_datatype: "float32"` for this first run so that thresholding can be performed directly in `ImageJ`, `Fiji`, or `napari` without additional rescaling; otherwise, when using `uint8` or `uint16`, the data must be converted to float32 before thresholding.

### Step 2: Run MC inference with xystd output

```yaml
# Step 2 config
output:
  save: ["xystd"]                # save both for overlay
model:
  num_mc: 5                            # adjust for accuracy vs runtime
  mc_threshold: -0.6                   # value chosen in Step 1
  tta_method: [null, "transpose"]      # at least one entry; richer lists give more TTA-driven variance
```

```bash
python test.py --env MyEnv --scale 8x --override MyExperiment
```

If `xystd` is in `output.save` but `mc_threshold` is `null`, the pipeline aborts at config-validation time before loading the model.

Total runtime scales linearly with `len(tta_method) × num_mc`. There is no benefit to making either extreme; modest values of both (e.g. 2 × 3 to 5 × 5) are usually enough for stable boundary maps.

## Result

- `xystd` is always saved as `uint8`

The figure below shows an example:
- Left: `xy` (model-enhanced output)
- Right: `xystd` (MC boundary map)

![MC Result Demo](assets/mc_result_demo.png)
