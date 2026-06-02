# semantic-demo

Companion engineering demo for the EE142 project.

## Goal
This is the secondary project track. It does **not** try to reproduce the theorems of the original paper directly. Instead, it builds a small, runnable demonstration of the paper's central intuition:

> when the receiver only cares about a task, the best transmitted representation need not preserve the full appearance of the raw observation.

## Why this track exists
The main theory track explains the rate-distortion viewpoint mathematically. This demo track provides an engineering-level illustration that is easier to show in a poster presentation or short oral presentation.

The planned comparison is:
1. **task-agnostic baseline**: compress or reduce the observation in a generic way, then classify;
2. **task-oriented representation**: learn a compact representation specifically optimized for the classification task.

If the task-oriented method achieves higher accuracy under the same communication budget, that visually supports the message of the original paper.

## Scope control
This track is intentionally small. It is supporting evidence, not the main theoretical contribution.

To keep it feasible, we start with a simple dataset such as MNIST or another lightweight classification dataset.

## Project structure
- `src/`: dataset handling and models
- `experiments/`: train/evaluate scripts
- `results/`: metrics and figures
- `docs/`: implementation rationale
