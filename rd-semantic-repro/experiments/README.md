# Experiments

Run from `~/zhuwl2022/it/rd-semantic-repro`.

## Verification

```bash
python3 -m pytest -q
```

## Binary classification reproduction scripts

```bash
python3 experiments/run_binary_curve.py
python3 experiments/plot_binary_curve.py
python3 experiments/plot_binary_curve_family.py
python3 experiments/plot_binary_soft_decision.py
python3 experiments/plot_binary_upper_bound.py
```

## Outputs

| Script | Purpose | Main output |
| --- | --- | --- |
| `run_binary_curve.py` | Write sampled `R(D_s, infinity)` points for `A=1, sigma=1` | `figures/binary_rd_infinite_a1_sigma1.csv` |
| `plot_binary_curve.py` | Plot Figure 4 right style `R(D_s, infinity)` curve | `figures/binary_rd_infinite_a1_sigma1.png` |
| `plot_binary_curve_family.py` | Compare `R(D_s, infinity)` under different class separations | `figures/binary_rd_infinite_curve_family.png` |
| `plot_binary_soft_decision.py` | Plot Figure 4 left style soft decision `g(x)` curves | `figures/binary_soft_decision_a1_sigma1.png` |
| `plot_binary_upper_bound.py` | Plot Figure 5 style Proposition 6 achievable upper bound | `figures/binary_upper_bound_a2sqrt10_sigma1.png` |