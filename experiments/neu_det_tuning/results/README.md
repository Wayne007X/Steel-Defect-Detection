# NEU-DET Experiment Archive

This directory contains archived experimental results for the steel surface defect detection project.

## Contents

- `raw_runs/`: archived CSV metrics, training arguments, error analysis, and experiment records.
- `logs/`: selected console logs from validation and controlled comparisons.
- `artifact_manifest.csv`: archive inventory with source paths, file sizes, SHA-256 hashes, and notes about sanitized files.

## Reproducibility notes

The archive preserves experimental evidence without requiring model weights or the original training outputs to be committed to this directory.

Some archived files have intentionally different hashes from their original source files because local absolute paths were replaced with portable placeholders. Consult the manifest notes when comparing hashes.

Evaluation metrics should be interpreted according to their recorded evaluation protocol. Standard validation metrics and fixed-confidence-threshold results are not interchangeable.
