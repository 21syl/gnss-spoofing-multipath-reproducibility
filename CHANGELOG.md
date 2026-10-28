# Changelog

## 0.2.0 - 2026-10-28

- Added the latest executable model and experiment modules in `src/gnss`.
- Aligned LSTM/CNN/ResNet1D/Transformer definitions with manuscript Table 3;
  exposed proposed-model component ablations and architecture descriptions.
- Added explicit raw-epoch MATLAB export, original-time offsets and train-only
  preprocessing; separated within-record and independent-record protocols.
- Added six training configurations, dataset parameter metadata and TEXBAT links.
- Added portable copies of SQM and DWT-fuzzy supplementary scripts, retaining
  their original algorithms and exposing record-specific configuration.
- Added five-model prediction timelines, unit tests and a synthetic smoke test.
- Each training run records input hashes, split assignments, normalization,
  environment, checkpoint, predictions and metrics. Original local scripts remain
  unchanged. Self-collected recordings and trained weights are not bundled.

## 0.0.0-demo - 2026-08-25

- Created the public-facing reproducibility repository scaffold.
- Added demo configuration, metadata, split, result, and documentation files.
- No real experimental data, trained weights, or validated results are included.
