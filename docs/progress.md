# Progress

## 2026-09-12

Shipped M1. `pytest tests/test_smoke.py` passed: one Conv1d on fake `(8, 12, 1000)` batches, BCEWithLogits, 20 steps, loss finite and last < first, device MPS.

Shipped M2. `python scripts/m2_audit.py` printed overlap 0, n 100, class counts NORM 42 / MI 24 / STTC 28 / CD 26 / HYP 9, 2 repeat patients, 24 co-occurring, wrote `figures/m2_lead2_grid.png`. Downloaded 100 Hz `.dat`/`.hea` only. `records500` is absent.

Next: M3 Dataset `{signal: (12,1000), label: (5,), lead_mask: (12,), patient_id}`, train-only norm, official folds. Do not train the full set in that step.
