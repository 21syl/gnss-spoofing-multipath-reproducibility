# Robust GNSS Spoofing Detection Under Multipath Interference

Reproducibility package for the manuscript "Robust GNSS spoofing detection under multipath interference: a multi-correlator Transformer-LSTM approach".

> **Demo repository scaffold.** The current files contain placeholders only. They must not be cited as released experimental data, executable research code, or reproduced results until the authors replace every `DEMO`, `PLACEHOLDER`, and `TODO` item.

## Planned contents

- `configs/`: versioned experiment configurations for the self-collected BDS B1I data and TEXBAT GPS L1 data.
- `data/`: dataset documentation, metadata dictionaries, checksums, and fixed record-level split manifests. Large data files are intentionally excluded from Git.
- `src/`: model, baseline, preprocessing, training, and evaluation source code.
- `scripts/`: one-command training, evaluation, and figure/table reproduction entry points.
- `results/`: machine-readable metrics, figure source data, tables, and logs.
- `docs/`: dataset card, availability statements, repository map, and release checklist.
- `third_party/TEXBAT/`: official access instructions and local preprocessing notes. TEXBAT recordings must not be redistributed unless their licence permits it.

## Intended reproducibility workflow

1. Obtain the self-collected dataset from the permanent repository listed in `docs/DATA_AVAILABILITY_DEMO.md`.
2. Obtain TEXBAT from its official source, then follow `third_party/TEXBAT/README.md`.
3. Create the software environment using `environment/environment.yml`.
4. Validate file integrity with `data/checksums/SHA256SUMS.demo.txt`.
5. Use the frozen manifests in `data/splits/` to prevent recording-window leakage.
6. Run the training and evaluation commands documented in `scripts/README.md`.
7. Regenerate paper figures and tables from machine-readable result files.

## Repository status

- Code: demo placeholders
- Self-collected data: not deposited
- TEXBAT: not redistributed
- Fixed split manifests: demo rows only
- Archived release DOI: not assigned

See `docs/UPDATE_GUIDE_ZH.md` for the Chinese update workflow.
