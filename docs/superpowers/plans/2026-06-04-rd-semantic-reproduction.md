# RD Semantic Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Complete the next reproduction layer for the semantic rate-distortion paper by adding Figure 4 g(x), Figure 5 Proposition 6 curves, and accurate run/theory documentation.

**Architecture:** Keep all numerical routines in d-semantic-repro/src/binary_case.py and add small experiment scripts under d-semantic-repro/experiments/. Tests encode theorem-level behavior before implementation. Generated CSV/PNG artifacts live under d-semantic-repro/figures/ and are tracked for report use.

**Tech Stack:** Python 3, pytest, matplotlib, csv, existing midpoint integration routines.

---

### Task 1: Expose soft-decision curve sampling for Figure 4 left panel

**Files:**
- Modify: d-semantic-repro/src/binary_case.py
- Modify: d-semantic-repro/tests/test_binary_case.py
- Create: d-semantic-repro/experiments/plot_binary_soft_decision.py

- [ ] **Step 1: Write failing tests**

Add tests importing sample_soft_decision_curve and checking symmetry, endpoints, and monotonicity.

- [ ] **Step 2: Verify RED**

Run: cd ~/zhuwl2022/it/rd-semantic-repro && python3 -m pytest tests/test_binary_case.py -q
Expected: FAIL because sample_soft_decision_curve is not defined.

- [ ] **Step 3: Implement minimal sampling function**

Implement sample_soft_decision_curve(A, sigma, Ds, x_min=-3.0, x_max=3.0, num_points=121) using _find_lambda and _compute_g.

- [ ] **Step 4: Add plot script**

Create experiments/plot_binary_soft_decision.py to generate igures/binary_soft_decision_a1_sigma1.csv and igures/binary_soft_decision_a1_sigma1.png for several D_s values matching the paper's Figure 4 style.

- [ ] **Step 5: Verify GREEN and commit**

Run pytest and the plot script. Commit tests, implementation, script, CSV, and PNG.

### Task 2: Add Figure 5 achievable upper-bound experiment

**Files:**
- Modify: d-semantic-repro/tests/test_binary_case.py
- Create: d-semantic-repro/experiments/plot_binary_upper_bound.py
- Track generated files under d-semantic-repro/figures/

- [ ] **Step 1: Write failing tests**

Add tests for sample_rd_finite_curve: positive finite rates for small D_a, lower or equal rates for larger D_a, and rejection of nonpositive D_a through d_upper_bound.

- [ ] **Step 2: Verify RED if a new behavior is missing**

Run the targeted tests. If all behavior already exists, keep tests as regression coverage and proceed to the missing experiment script.

- [ ] **Step 3: Add plot script**

Create experiments/plot_binary_upper_bound.py that samples rate vs D_a for D_s values 0.50, 0.30, and a feasible low-distortion value for A^2/sigma^2 = 10, plus the two baseline curves shown in the paper.

- [ ] **Step 4: Run script and verify outputs**

Run script and confirm CSV/PNG are created and nonempty.

- [ ] **Step 5: Run full tests and commit**

Run python3 -m pytest -q, commit the new tests, script, and figures.

### Task 3: Update reproduction documentation

**Files:**
- Modify: d-semantic-repro/README.md
- Modify: d-semantic-repro/experiments/README.md
- Modify: d-semantic-repro/src/README.md
- Modify: d-semantic-repro/docs/implementation-notes.md

- [ ] **Step 1: Correct stale status**

Remove statements that the Proposition 5 interior formula is staged or pending.

- [ ] **Step 2: Add exact reproduction commands**

Document dependency installation, pytest command, and every experiment script command.

- [ ] **Step 3: Map scripts to paper figures**

List Figure 4 right, Figure 4 left, class-separation family, and Figure 5 upper-bound outputs.

- [ ] **Step 4: Remove nonexistent-file claims**

Keep Gaussian and multiclass items as future work, not implemented structure.

- [ ] **Step 5: Verify and commit**

Run tests and scripts again, then commit documentation updates.

### Task 4: Final verification checkpoint

**Files:**
- No code changes expected.

- [ ] **Step 1: Run full test suite**

Run: cd ~/zhuwl2022/it/rd-semantic-repro && python3 -m pytest -q
Expected: all tests pass.

- [ ] **Step 2: Regenerate all figures**

Run all experiment scripts and verify no unexpected git diff except intended tracked outputs.

- [ ] **Step 3: Confirm git status**

Run: cd ~/zhuwl2022/it && git status --short && git log --oneline -5
Expected: clean status and recent reproduction commits visible.
