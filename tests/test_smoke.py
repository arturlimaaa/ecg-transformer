import math

from ecg_transformer.smoke import train_synthetic


def test_synthetic_cnn_loss_is_finite_and_falls_over_twenty_steps():
    losses = train_synthetic(steps=20, batch_size=8, seed=0)
    assert len(losses) == 20
    assert all(math.isfinite(v) for v in losses)
    assert losses[-1] < losses[0]
