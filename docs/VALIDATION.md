# Implementation validation

Validation on 2026-10-28 used Python 3.12.9, PyTorch 2.3.1+cu118,
NumPy 1.26.4, pandas 2.2.3, h5py 3.14.0 and scikit-learn 1.6.1.

Seven unit tests passed, covering architecture specifications, forward/backward
passes for two/three classes, evaluation-time batch independence, boundary-safe
windows, seeded stratification, training-only normalization, absent-class metrics
and HDF5 transfer loading. The five-model CPU smoke test completed training,
checkpoint selection and evaluation on synthetic arrays.

A synthetic MATLAB v7.3 fixture was exported using channel 8 and an explicit
99 s original-time offset. Its 100/101 s event boundaries, 11 selected taps,
HDF5 readback and Python [3000,11] layout were checked. These are software tests,
not GNSS experimental observations or accuracy estimates. The original local
experiment files were not modified. Real-data execution requires the input
recordings and completed local record mappings.
