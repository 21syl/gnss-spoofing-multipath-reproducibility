# Reproducibility release checklist

- [ ] Replace every `DEMO`, `PLACEHOLDER`, and `TODO` marker.
- [ ] Confirm code and data ownership with all contributors and the institution.
- [ ] Select compatible code and data licences.
- [ ] Freeze record-level or scenario-level train, validation, and test manifests.
- [ ] Verify that normalization is fitted using training data only.
- [ ] Pin Python, CUDA, framework, and package versions.
- [ ] Add all model and baseline implementations.
- [ ] Add deterministic training and evaluation commands.
- [ ] Add figure and table source data plus generation scripts.
- [ ] Add model parameter count, FLOPs, memory, and hardware information when reported.
- [ ] Add checksums for deposited data and trained checkpoints.
- [ ] Test the workflow in a clean environment.
- [ ] Deposit data in a durable repository and record its DOI or accession.
- [ ] Archive the manuscript-specific GitHub release in Zenodo and record its DOI.
- [ ] Update `CITATION.cff`, README status, manuscript availability text, and repository topics.
- [ ] Test public and reviewer links outside the author account.
