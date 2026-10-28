# GNSS spoofing detection under multipath interference

Code and experimental dataset information for **Robust GNSS spoofing detection
under multipath interference: a multi-correlator Transformer-LSTM approach**.

**Publication status:** The associated manuscript is currently under peer review
and has not yet been accepted for publication.

The repository contains the proposed model, CNN, ResNet1D, LSTM, Transformer,
sliding Delta, WSCM and DWT-fuzzy, together with data processing, training,
evaluation and plotting code.

## Repository structure

* `code/src/gnss/`: neural models, preprocessing, training and evaluation.
* `code/configs/experiments/`: training settings and experiment configurations.
* `code/scripts/run_experiment.py`: Python training and evaluation entry point.
* `code/matlab/`: correlator export, SQM/DWT-fuzzy methods and result plotting.
* `dataset/configs/`: signal parameters, recording configurations and time alignment.

## Installation and execution

Run the commands from the repository root. The code uses Python 3.10+ and MATLAB.
DWT-fuzzy additionally requires MATLAB Wavelet Toolbox.

```sh
python -m pip install -r code/requirements.txt
```

Set the local MAT paths, tracking channels, processed durations and time offsets
in `dataset/configs/`. Export the selected correlator epochs in MATLAB:

```matlab
addpath('code/matlab');
export_correlator_records('dataset/configs/mixed_records.json');
```

Train and evaluate the two mixed-record transfer experiments:

```sh
python code/scripts/run_experiment.py --config code/configs/experiments/mixed_static_to_dynamic.json --data-dir dataset/processed/mixed
python code/scripts/run_experiment.py --config code/configs/experiments/mixed_dynamic_to_static.json --data-dir dataset/processed/mixed
```

Other configurations cover pairwise static/dynamic tests and TEXBAT
within-scenario/cross-scenario tests. Source and target records, model selection,
normalization and training settings are configured in JSON. Use `--models cnn lstm`
to select models or `--device cpu` to run on the CPU.

Each run saves its configuration, partitions, fitted normalization, checkpoint,
training history, predictions and metrics in a timestamped directory under
`outputs/`. Metrics include accuracy, precision, recall, F1 and AUC, with additional
stage-specific metrics for mixed signals.

Plot five-model transfer predictions by supplying the two output directories:

```matlab
plot_prediction_timelines({runA,runB},{'bs6','bs7'},'outputs/figures');
```

For the MATLAB baselines, specify local paths and experiment settings:

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

## Model and training configurations

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
epochs. Testing uses the minimum-validation-loss checkpoint. Component ablations
are configured through the model names in `code/configs/experiments/`.

## Experimental datasets

The experiments use self-collected BDS B1I recordings and the public TEXBAT GPS
L1 C/A dataset. Self-collected recordings are available upon reasonable request
by email to the corresponding author listed in the paper. The recordings are
transferred separately because of their file size.

### Self-collected recordings

The signal settings are BDS B1I, authentic C/N0 of 45 dB·Hz, spoofing power
advantage of 3 dB and asynchronous code pull-off at 0.1 chips/s from -4 to 4 chips.
Dynamic recordings use a platform speed of 15 m/s. Each experiment processes one
tracked satellite. Multipath generation follows the models cited in the paper
([reference 1](https://doi.org/10.1109/TVT.2020.3038646),
[reference 2](https://doi.org/10.1186/s43020-020-00027-7)).

| Record | Platform | Signal sequence |
|---|---|---|
| bs1 | Static | Authentic → authentic + spoofing |
| bs2 | Dynamic | Authentic → authentic + spoofing |
| bs3 | Static | Authentic → authentic + multipath |
| bs4 | Dynamic | Authentic → authentic + multipath |
| bs5 | Static | Authentic → authentic + multipath → authentic + multipath + spoofing |
| bs6 | Dynamic | Authentic → authentic + multipath → authentic + multipath + spoofing |
| bs7 | Static | Authentic → authentic + spoofing → authentic + spoofing + multipath |
| bs8 | Dynamic | Authentic → authentic + spoofing → authentic + spoofing + multipath |

The original pairwise signals last 200 s, with interference injected at 100 s.
Receiver-output selections are aligned using the original time of the first
selected epoch. Mixed records contain 198 s of processed epochs on channel 8.
Their first and second interference types enter at 80 and 120 s, measured from
the first tracking epoch; the first interference remains present afterward.

### TEXBAT

Obtain TEXBAT from the [UT Austin Radionavigation Laboratory](https://radionavlab.ae.utexas.edu/texbat/)
or its [official download directory](https://rnl-data.ae.utexas.edu/datastore/texbat/).
The cross-scenario experiment uses ds2/ds4/ds5 for training and ds3/ds6/ds7/ds8
for testing. Receiver processing produces the correlator MAT inputs used by the
export script.

### Input construction and partitions

Neural-network inputs consist of I-branch amplitudes at 11 correlators over
[-2,2] chips, with positions -2, -1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6
and 2 chips. Each sample contains ten 1 ms epochs, with a 10 ms window and step.
The tensor layout is [sample, correlator, time]. Labels are 0 for authentic only,
1 for spoofing present (including simultaneous multipath), and 2 for multipath
without spoofing. Event timestamps determine the labels; windows stay within
their signal-state intervals.

Pairwise and TEXBAT within-scenario tests partition each continuous signal-state
interval chronologically at 60/20/20 before constructing windows. Classes are
balanced by sampling within each partition. Standardization is fitted on training
samples and applied to validation and test samples.

Mixed-record transfer trains on bs5 and tests bs6, or trains on bs8 and tests bs7.
Source windows are split by class-stratified sampling at 70/15/15. One min/max
pair is fitted on source training samples and applied to the remaining samples.
TEXBAT transfer uses an 80/20 source training/validation split and training-fitted
standardization. Independent target records are used for testing.

Detailed signal parameters and record identifiers are in
`dataset/configs/dataset_parameters.json`. MATLAB reads
`trackData.<system>.channel` and exports selected raw correlator epochs to HDF5;
Python constructs the windows and fits the configured normalization.
