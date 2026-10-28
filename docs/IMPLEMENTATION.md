# Implementation and experimental protocols

This is the latest code version. Network definitions follow manuscript Table 3
and Section IV. The original local experiment scripts are not overwritten.

## Architectures

| Model | Configuration |
|---|---|
| CNN | Three Conv1D blocks, kernels 3, widths 64/128/256; batch normalization, ReLU and max pooling in each block |
| LSTM | Two recurrent layers, 128 hidden units per layer, final time-step classification |
| ResNet1D | Four residual blocks, kernels 3, widths 64/128/256/512 |
| Transformer | Four encoder layers, embedding 256, eight heads, feed-forward width 512, sinusoidal positional encoding |
| Proposed | Spatial convolution, three temporal scales, two-layer bidirectional LSTM, multi-head attention and classifier |

The source defines every operation. Details not specified by Table 3 are
implementation choices: CNN/ResNet/Transformer global-mean readout, two convolutions
per residual block with projected skips, Transformer dropout 0.1 and sinusoidal
encoding. Proposed uses 64/128 spatial channels, 64 channels per temporal branch,
256 recurrent units per direction and eight attention heads. Interpolation is
linear along the one-dimensional time axis. These choices are not attributed to
unavailable reference implementations. Ablation names are `no_spatial`,
`no_multiscale` and `no_spatial_no_multiscale`.

## Input and time alignment

MATLAB exports signed I epochs without fitting statistics. Python uses abs(I),
producing [sample, correlator, time] tensors with 11 by 10 entries. The 11 positions
are -2, -1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6 and 2 chips on the original
81-tap grid. Other odd counts select the nearest grid positions and save them.
Each epoch is 1 ms. Windows are 10 ms long with a 10 ms step.

For pairwise signals, interference begins at original signal time 100 s. A
receiver output may start later. Set `processed_start_original_s` to the original
time of the first **selected** epoch; set `read_start_epoch` if selecting a later
epoch from the MAT file. The event index is computed from these explicit times.
The code does not infer an offset from a filename or silently skip data. The
pairwise JSON is a mapping template: file, channel, offset and available duration
must be supplied. The nominal 200 s signal duration is not a promise of 200 s of
processed epochs. For noncontiguous selections, export separate records with
their own offsets rather than treating them as a continuous timeline.

Mixed recordings use channel 8, 198 s and times relative to their first tracking
epoch. Multipath-first recordings have events at 80/120 s; spoofing-first records
reverse the order. The first interference persists. Physical states are
0=authentic, 1=authentic+spoofing, 2=authentic+multipath, 3=all three. Classification
labels are 0/1/2; state 3 has label 1 because spoofing is present.

## Splits and preprocessing

* Pairwise and TEXBAT within-record experiments split each continuous physical-state
  interval chronologically 60/20/20 **before** creating windows. No window crosses
  an event or split boundary. Incomplete final windows are dropped.
* Mixed transfer uses bs5 (static multipath-first) to bs6 (dynamic multipath-first),
  and bs8 (dynamic spoofing-first) to bs7 (static spoofing-first). Nonoverlapping
  source windows are stratified 70/15/15 by class. The independent target record
  is not involved in training or selection. Source holdout and target results
  are reported separately. These source random splits are not independent-record
  estimates; target evaluation is the independent-record test.
* TEXBAT transfer trains on ds2/ds4/ds5 and tests ds3/ds6/ds7/ds8. Source windows
  are stratified 80/20 for training/validation, an explicit configuration choice.
* Pairwise/TEXBAT balance classes by seeded subsampling **within each partition**,
  without augmentation. Mixed transfer retains its natural stage proportions.
* Pairwise/TEXBAT fit per-correlator z-score statistics on selected training
  windows only. Mixed transfer fits one global min/max on source training windows
  only. Validation, source holdout and targets reuse those values. Targets are
  not clipped and may lie outside [0,1]; no target statistics are fitted.

## Optimization and selection

All configurations use Adam, cross-entropy, seed 42 and zero weight decay.
Pairwise/TEXBAT use batch 64, learning rate 0.001, a tenfold reduction every ten
epochs, at most 50 epochs and validation-loss patience 10. Mixed transfer uses
batch 32, learning rate 0.001, no scheduler and at most ten epochs. These settings
are editable JSON values. Each model restores its minimum-validation-loss
checkpoint. Test results never select checkpoints. PyTorch reproducibility is
hardware/version dependent; environments and seeds are recorded, not a guarantee
of bitwise agreement across platforms.

## Metrics and stored results

Accuracy counts correct window labels. CSVs contain window start/end, physical
state, true label, prediction and class probabilities. Overall and physical-stage
metrics are separate. Spoofing detection is prediction==1 among spoofing-present
windows; multipath false alarms are prediction==1 among state-2 windows. Macro
precision/recall/F1 use classes present in the evaluated true labels; the full
macro AUC is undefined when a class is absent and is left empty. Present-class
one-vs-rest AUC and binary spoofing AUC have distinct names.

Each run saves configuration, input SHA-256 hashes, record metadata, selected and
discarded window assignments, class counts, preprocessing statistics, environment,
architecture, parameter count, checkpoint, learning curves, predictions and metrics.
No precomputed manuscript scores or checkpoints are included in this release.

## MATLAB SQM and fuzzy implementations

`matlab/baselines/` contains portable copies of the existing supplementary scripts.
Their calibration intervals, smoothing, tap choices and fuzzy membership rules
are preserved, not silently replaced by the neural-network protocol. Pass a
configuration struct to override all record-specific settings. All times in these
legacy scripts refer to their receiver-output timeline; adjust for the actual
recording offset. File paths are relative to the repository root.

The supplied SQM preset has different smoothing windows across its two records;
it must not be described as a common-window detector without setting the two
window parameters equal. The fuzzy code uses 81-point |I+jQ|, not the neural
I-only input. Its current selected coefficient list is
[17 18 20 21 27 38 39 40]; membership functions are retained as provided. These
details are explicit so that settings can be checked before numerical comparison.
Fuzzy requires Wavelet Toolbox. Both scripts export their features, predictions,
metrics and editable figures. They are project implementations, not verbatim
third-party author code.

## Figure coverage

`matlab/plot_prediction_timelines.m` plots all five models directly from trainer
CSVs, with stage accuracy and no smoothing or downsampling. SQM/fuzzy scripts
include feature and result plots. These are result-plotting tools; conceptual
diagrams and all original manuscript figure layouts are not regenerated here.
The legacy files named `demo` remain scaffold examples, not experimental evidence.
