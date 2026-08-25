# Data directory

This directory currently contains demo metadata only. Do not commit large raw or processed data directly to Git.

## Planned access routes

- Self-collected BDS B1I data: deposit the processed multi-correlator inputs, labels, metadata, and split manifests in a durable repository such as Zenodo, then record its DOI in `docs/DATA_AVAILABILITY_DEMO.md`.
- Raw RF/IQ recordings: document the public or controlled access route and any genuine restriction.
- TEXBAT: retain only official access instructions, file checksums, local file manifests, and preprocessing code unless redistribution rights are confirmed.

## Required provenance

Each released sample should map to a recording, scenario, platform state, PRN or satellite identifier, class label, time interval, processing version, and exactly one train, validation, or test subset.
