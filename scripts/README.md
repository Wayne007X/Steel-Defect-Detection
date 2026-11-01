# Research Scripts

This directory organizes the project's research code by purpose.

- `training/`: model training and fine-tuning experiments.
- `evaluation/`: model evaluation, controlled comparisons, and validation.
- `analysis/`: dataset label audits and prediction error analysis.
- `postprocessing/`: confidence threshold sweeps and post-processing experiments.

These scripts document the experimental workflow. Some scripts may contain paths configured for the original Windows workstation. Review the dataset and model paths before running them on another machine.

The trained model weights and local training outputs are not included in this directory.

## Current fixed-protocol entry points

- [Code and configuration catalog](../docs/CODE_AND_CONFIG.md): canonical rebuilt research and local web entry points.
- [Read-only evidence verifier](testing/verify_repository_evidence.py): repository scope uses Python standard library only; no training/inference/Test release.
- Existing scripts in this directory remain historical research records unless the catalog says otherwise.
