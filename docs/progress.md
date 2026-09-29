# Progress

## 2026-09-12

Shipped M1. `pytest tests/test_smoke.py` passed: one Conv1d on fake `(8, 12, 1000)` batches, BCEWithLogits, 20 steps, loss finite and last < first, device MPS.

Shipped M2. `python scripts/m2_audit.py` printed overlap 0, n 100, class counts NORM 42 / MI 24 / STTC 28 / CD 26 / HYP 9, 2 repeat patients, 24 co-occurring, wrote `figures/m2_lead2_grid.png`. Downloaded 100 Hz `.dat`/`.hea` only. `records500` is absent.

Shipped M3. Full `records100` tree is 21799 `.dat` (596M), `records500` absent. Official folds 17418/2183/2198, patient sets disjoint by set intersection. 411 unlabeled rows kept as zeros. Diagnostic codes with likelihood 0 still count (PhysioNet rule). Packed train-only z-scored memmap `data/ptb-xl/packed/signals.npy` (998M). `pytest tests` 23 passed. Dataset returns `signal (12,1000)`, `label (5,)`, `lead_mask (12,)` ones, `patient_id`, `ecg_id`.

Shipped M4. Train-frequency baseline on fold 10: per-class scores are train positive rates (NORM 0.436 / MI 0.251 / STTC 0.240 / CD 0.224 / HYP 0.122). Fold-10 AUROC is 0.5 for every class, macro-AUROC 0.5, macro-AUPRC 0.254, n_test 2198. Hard majority is all zeros (every rate < 0.5). `pytest tests` 27 passed. Wrote `figures/m4_metrics.json`.

Shipped M5. 1D CNN, unweighted BCEWithLogits, Adam lr 1e-3, batch 128, seed 0, max 25 epochs, patience 5. Checkpoint is epoch 23 by fold-9 macro-AUROC 0.926. Fold 10 scored once, clean leads: macro-AUROC 0.920, macro-AUPRC 0.803, per-class AUROC NORM 0.944 / MI 0.921 / STTC 0.929 / CD 0.913 / HYP 0.891, n_test 2198. Wrote `figures/m5_metrics.json`, `figures/m5_curves.png`, `checkpoints/m5_cnn.pt`.

Shipped M7. Same M5 checkpoint, no retraining, fold 10, rows not averaged. Clean macro-AUROC 0.920 matches `figures/m5_metrics.json`. V1–V3 off: macro-AUROC 0.875 (MI 0.836, HYP 0.874). I+II only: macro-AUROC 0.721 (MI 0.571, HYP 0.665). Wrote `figures/missing_lead_table.json`.

Shipped M8. Demo is fold-10 `ecg_id` 514, the first MI-only test record with clean MI probability at least 0.8. Sigmoid probabilities: clean MI 0.819 / CD 0.692 / HYP 0.429; V1–V3 off MI 0.795 / CD 0.856 / HYP 0.261; I+II only MI 0.987 / CD 0.487 / HYP 0.002. The three vectors differ. Wrote `figures/m8_demo.json` and `figures/m8_demo.png`.

Shipped M6. Supervised Transformer, no pretraining, no reconstruction head. Patches of 8 samples (80 ms) at 100 Hz, d_model 64, 2 layers, 4 heads, dropout 0. Same loss, split, and fold-9 selection as M5. Stopped at epoch 17; checkpoint is epoch 12, fold-9 macro-AUROC 0.907. Fold 10 clean macro-AUROC 0.895, macro-AUPRC 0.763, per-class AUROC NORM 0.925 / MI 0.883 / STTC 0.918 / CD 0.858 / HYP 0.892. Same three masks, no retraining: V1–V3 off macro-AUROC 0.837, I+II only 0.641. Wrote `figures/m6_metrics.json`, `figures/m6_curves.png`, `checkpoints/m6_transformer.pt`, and added the block to `figures/missing_lead_table.json`.

The CNN remains ahead on every row (clean 0.920 / 0.875 / 0.721). Course track A is complete.
