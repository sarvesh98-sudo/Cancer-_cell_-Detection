"""
Train a bootstrap CNN with varied synthetic H&E patches.

Labels follow nuclear density (sparse = No, dense = Yes), with brightness /
stain / noise augmentation so real uploads are less likely to collapse to one class.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from model_utils import MODEL_PATH, create_advanced_model

RNG = np.random.default_rng(42)
IMG = 96
N_TRAIN = 3000
N_VAL = 600
EPOCHS = 5
BATCH = 32


def make_patch(label: int) -> np.ndarray:
    """Synthetic 96x96 histopathology-like RGB patch."""
    # Vary overall stain brightness so the model doesn't lock onto one pink tone
    base = RNG.normal(loc=[200, 150, 170], scale=18, size=(IMG, IMG, 3))
    base *= RNG.uniform(0.75, 1.15)

    if label == 0:
        n_blobs = int(RNG.integers(4, 22))
        intensity = float(RNG.uniform(0.35, 0.65))
        radius = (2, 4)
    else:
        n_blobs = int(RNG.integers(28, 85))
        intensity = float(RNG.uniform(0.65, 0.95))
        radius = (2, 7)
        cy, cx = RNG.integers(28, IMG - 28, size=2)
        yy, xx = np.ogrid[:IMG, :IMG]
        wash = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 20**2))
        base[..., 0] -= 45 * wash
        base[..., 2] += 30 * wash

    # Bias blobs toward center for positive class (PCam-like)
    for _ in range(n_blobs):
        if label == 1 and RNG.random() < 0.7:
            cy = int(RNG.integers(28, 68))
            cx = int(RNG.integers(28, 68))
        else:
            cy = int(RNG.integers(0, IMG))
            cx = int(RNG.integers(0, IMG))
        r = int(RNG.integers(*radius))
        yy, xx = np.ogrid[:IMG, :IMG]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
        base[mask, 0] *= 1 - intensity * 0.75
        base[mask, 1] *= 1 - intensity * 0.55
        base[mask, 2] = np.clip(
            base[mask, 2] * (1 - intensity * 0.15) + 45 * intensity, 0, 255
        )

    base = np.clip(base + RNG.normal(0, 6, base.shape), 0, 255).astype(np.float32) / 255.0
    return base


def build_dataset(n: int):
    labels = RNG.integers(0, 2, size=n)
    images = np.stack([make_patch(int(y)) for y in labels], axis=0)
    return images, labels.astype(np.float32)


def main():
    print("Generating synthetic patches…")
    x_train, y_train = build_dataset(N_TRAIN)
    x_val, y_val = build_dataset(N_VAL)

    model = create_advanced_model()
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print("Training bootstrap model…")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        verbose=1,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    acc = history.history["val_accuracy"][-1]
    print(f"Saved → {MODEL_PATH}")
    print(f"Final val accuracy: {acc:.1%}")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
