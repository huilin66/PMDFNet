# Tools

`tools/` contains code that supports the current PMDFNet research workflow but
is not part of the paper training entry point.

- `data/map_mmlsv2_to_mars.py`: reproducible MMLSv2 → Mars-LS representation mapping.
- `evaluation/`: frozen-checkpoint validation/test evaluation. These scripts
  must only be used after model and checkpoint selection are frozen.
  This includes `cam_analysis.py` (Grad-CAM),
  `tsne_prototype_analysis.py` (GT/Pred feature prototypes) and
  `make_cam_comparison_figure.py` (inference-free figure composition), plus
  `evaluate_test_diagnostics.py` for boundary metrics, per-tile distributions,
  foreground-area strata and predictive-entropy diagnostics.
- `validation/`: data-only smoke checks and bounded synthetic VRAM probes.

Historical competition, SKG, submission, and analysis code is preserved under
[`archive/legacy/`](../archive/legacy/); it is not imported by the current
paper protocol.
