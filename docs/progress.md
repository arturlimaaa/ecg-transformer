# Progress

## 2026-09-12

Shipped M1. `pytest tests/test_smoke.py` passed: one Conv1d on fake `(8, 12, 1000)` batches, BCEWithLogits, 20 steps, loss finite and last < first, device MPS.

Shipped M2. `python scripts/m2_audit.py` printed overlap 0, n 100, class counts NORM 42 / MI 24 / STTC 28 / CD 26 / HYP 9, 2 repeat patients, 24 co-occurring, wrote `figures/m2_lead2_grid.png`. Downloaded 100 Hz `.dat`/`.hea` only. `records500` is absent.

Shipped M3. Full `records100` tree is 21799 `.dat` (596M), `records500` absent. Official folds 17418/2183/2198, patient sets disjoint by set intersection. 411 unlabeled rows kept as zeros. Diagnostic codes with likelihood 0 still count (PhysioNet rule). Packed train-only z-scored memmap `data/ptb-xl/packed/signals.npy` (998M). `pytest tests` 23 passed. Dataset returns `signal (12,1000)`, `label (5,)`, `lead_mask (12,)` ones, `patient_id`, `ecg_id`.

Shipped M4. Train-frequency baseline on fold 10: per-class scores are train positive rates (NORM 0.436 / MI 0.251 / STTC 0.240 / CD 0.224 / HYP 0.122). Fold-10 AUROC is 0.5 for every class, macro-AUROC 0.5, macro-AUPRC 0.254, n_test 2198. Hard majority is all zeros (every rate < 0.5). `pytest tests` 27 passed. Wrote `figures/m4_metrics.json`.

Next: M5 CNN on fold 10. Early-stop on fold 9. Must beat this 0.5 macro-AUROC.
