# Algorithms

The Python implementation provides the proposed Transformer–LSTM model, CNN,
ResNet1D, LSTM and Transformer baselines, and the proposed model's component
ablations. MATLAB scripts export correlator inputs, run sliding Delta/WSCM and
DWT-fuzzy, and plot detection results.

## Run

Run the commands below from the repository root. Python 3.10+ and MATLAB are used.
DWT-fuzzy additionally requires MATLAB Wavelet Toolbox.

```sh
python -m pip install -r code/requirements.txt
```

Configure local MAT paths, channels and time references in `dataset/configs/`.
Export the selected correlator epochs in MATLAB:

```matlab
addpath('code/matlab');
export_correlator_records('dataset/configs/mixed_records.json');
```

Train and evaluate:

```sh
python code/scripts/run_experiment.py --config code/configs/experiments/mixed_static_to_dynamic.json --data-dir dataset/processed/mixed
python code/scripts/run_experiment.py --config code/configs/experiments/mixed_dynamic_to_static.json --data-dir dataset/processed/mixed
```

The other configurations cover pairwise static/dynamic tests and TEXBAT
within-scenario/cross-scenario tests. Source and target recordings, training
settings and model selection are configured in JSON. Use `--models cnn lstm`
to select models and `--device cpu` to use the CPU.

Each run saves its configuration, data partitions, fitted normalization,
checkpoint, training history, predictions and metrics in a timestamped directory
under `outputs/`. Accuracy, precision, recall, F1 and AUC are calculated from saved
test predictions; mixed-signal results also include metrics for each physical stage.

Plot the five-model transfer results using the two run directories:

```matlab
plot_prediction_timelines({runA,runB},{'bs6','bs7'},'outputs/figures');
```

For the MATLAB baselines, pass local paths and experiment settings as a struct:

```matlab
addpath('code/matlab/baselines');
cfg = struct('spoofFile','dataset/raw/spoofing.mat', ...
    'multipathFile','dataset/raw/multipath.mat');
sqmResults = sqm_20260922_delt_wscm(cfg);
fuzzyResults = run_fuzzy_dwt_threeclass(cfg);
```

Calibration intervals, event times, correlator positions, smoothing settings and
output directories are specified in each baseline's `defaultConfig` and can be
overridden through the input struct. The scripts save intermediate features,
metrics and editable MATLAB figures.

## Network configurations

| Model | Configuration |
|---|---|
| CNN | Three Conv1D blocks, kernel 3, channels 64/128/256, batch normalization, ReLU and max pooling |
| LSTM | Two LSTM layers, 128 hidden units per layer |
| ResNet1D | Four residual blocks, kernel 3, channels 64/128/256/512 |
| Transformer | Four encoder layers, embedding 256, eight heads, feed-forward width 512, positional encoding |
| Proposed | Spatial convolution, three temporal branches, two-layer bidirectional LSTM, multi-head attention and classifier |

Pairwise experiments use Adam, learning rate 0.001 reduced tenfold every ten
epochs, batch size 64, cross-entropy loss, patience 10, at most 50 epochs and seed
42. Mixed-record training uses batch size 32, learning rate 0.001 and at most ten
epochs. The minimum-validation-loss checkpoint is used for testing.
