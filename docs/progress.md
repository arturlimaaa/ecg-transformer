# Progress

## 2026-09-12

Shipped M1. `pytest tests/test_smoke.py` passed: one Conv1d on fake `(8, 12, 1000)` batches, BCEWithLogits, 20 steps, loss finite and last < first, device MPS.

Shipped M2. `python scripts/m2_audit.py` printed overlap 0, n 100, class counts NORM 42 / MI 24 / STTC 28 / CD 26 / HYP 9, 2 repeat patients, 24 co-occurring, wrote `figures/m2_lead2_grid.png`. Downloaded 100 Hz `.dat`/`.hea` only. `records500` is absent.

Shipped M3. Full `records100` tree is 21799 `.dat` (596M), `records500` absent. Official folds 17418/2183/2198, patient sets disjoint by set intersection. 411 unlabeled rows kept as zeros. Diagnostic codes with likelihood 0 still count (PhysioNet rule). Packed train-only z-scored memmap `data/ptb-xl/packed/signals.npy` (998M). `pytest tests` 23 passed. Dataset returns `signal (12,1000)`, `label (5,)`, `lead_mask (12,)` ones, `patient_id`, `ecg_id`.

Next: M4 majority baseline on fold 10. Do not train the CNN in that step.
