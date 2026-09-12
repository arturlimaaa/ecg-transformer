# Progress

## 2026-09-12

Shipped M1. `pytest tests/test_smoke.py` passed: one Conv1d on fake `(8, 12, 1000)` batches, BCEWithLogits, 20 steps, loss finite and last < first, device MPS.

Next: M2 100-record audit. Download `records100` plus metadata only. Do not fetch `records500`. Debug set must not be all NORM.
