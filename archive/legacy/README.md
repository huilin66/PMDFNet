# Legacy archive

This directory preserves code and documents from the competition/reproduction
workflow and the earlier SKG-PMDFNet branch. It is retained for traceability
and old-checkpoint investigation, but it is not part of the current PMDFNet
paper protocol.

- `root/`: historical training entry points and SKG modules.
- `configs/`: superseded ablation and descriptor configurations.
- `scripts/`: old competition, fusion, capacity, and SKG launchers.
- `tools/`: historical result analysis, submission generation, and SKG tools.
- `docs/`: superseded handoff and experiment-plan documents.

The active workflow is defined by `configs/mmlsv2_paper_protocol.json`,
`train_ablation.py`, `scripts/train_paper_experiments.*`, and the current
`tools/data`, `tools/evaluation`, and `tools/validation` directories.
