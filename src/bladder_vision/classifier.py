"""CNN training and inference for frame-level bladder lesion classification."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .constants import (
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
)


def build_cnn(
    num_classes: int,
    *,
    input_shape: tuple[int, int, int] = (
        DEFAULT_INPUT_HEIGHT,
        DEFAULT_INPUT_WIDTH,
        3,
    ),
) -> Any:
    """Build the compact convolutional network used by the reference pipeline."""
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255.0, input_shape=input_shape),
            tf.keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
            tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="bladder_cnn",
    )
    return model


def _make_dataset(
    directory: Path,
    *,
    height: int,
    width: int,
    batch_size: int,
    seed: int,
    shuffle: bool,
    validation_split: float | None = None,
    subset: str | None = None,
) -> Any:
    import tensorflow as tf

    kwargs: dict[str, Any] = {
        "directory": str(directory),
        "labels": "inferred",
        "label_mode": "categorical",
        "image_size": (height, width),
        "batch_size": batch_size,
        "shuffle": shuffle,
        "seed": seed,
    }
    if validation_split is not None:
        kwargs["validation_split"] = validation_split
    if subset is not None:
        kwargs["subset"] = subset
    return tf.keras.utils.image_dataset_from_directory(**kwargs)


def train_classifier(
    *,
    data_dir: str | Path,
    output_path: str | Path,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    height: int = DEFAULT_INPUT_HEIGHT,
    width: int = DEFAULT_INPUT_WIDTH,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = DEFAULT_SEED,
    validation_split: float = 0.2,
    patience: int = 5,
    steps_per_epoch: int | None = None,
) -> dict[str, Any]:
    """Train the reference CNN on a directory of class folders.

    The expected layout is ``data/train/<class>`` and optionally
    ``data/validation/<class>``. If a validation directory is absent, a seeded
    fraction of the training set is held out. Class names are written next to the
    model as ``<model>.classes.json`` so inference cannot silently reorder them.
    """
    import tensorflow as tf

    data_path = Path(data_dir)
    train_dir = data_path / "train"
    validation_dir = data_path / "validation"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    if validation_dir.is_dir():
        train_ds = _make_dataset(
            train_dir,
            height=height,
            width=width,
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
        )
        validation_ds = _make_dataset(
            validation_dir,
            height=height,
            width=width,
            batch_size=batch_size,
            seed=seed,
            shuffle=False,
        )
        class_names = list(train_ds.class_names)
    else:
        train_ds = _make_dataset(
            train_dir,
            height=height,
            width=width,
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
            validation_split=validation_split,
            subset="training",
        )
        validation_ds = _make_dataset(
            train_dir,
            height=height,
            width=width,
            batch_size=batch_size,
            seed=seed,
            shuffle=False,
            validation_split=validation_split,
            subset="validation",
        )
        class_names = list(train_ds.class_names)

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.02),
            tf.keras.layers.RandomZoom(0.05),
        ],
        name="augmentation",
    )
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(
        lambda images, labels: (augmentation(images, training=True), labels),
        num_parallel_calls=autotune,
    ).prefetch(autotune)
    validation_ds = validation_ds.prefetch(autotune)
    if steps_per_epoch:
        train_ds = train_ds.repeat()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = build_cnn(len(class_names), input_shape=(height, width, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(destination),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
    )
    model.save(destination)

    classes_path = destination.with_suffix(".classes.json")
    classes_path.write_text(json.dumps(class_names, indent=2) + "\n", encoding="utf-8")
    return {
        "model_path": str(destination),
        "classes_path": str(classes_path),
        "class_names": class_names,
        "history": history.history,
    }


def load_classifier(model_path: str | Path) -> Any:
    """Load a Keras model saved by :func:`train_classifier` or the legacy scripts."""
    import tensorflow as tf

    return tf.keras.models.load_model(str(model_path))


def load_class_names(model_path: str | Path, fallback: Sequence[str] = CLASS_NAMES) -> list[str]:
    """Read the class-name sidecar when present, otherwise use the canonical order."""
    sidecar = Path(model_path).with_suffix(".classes.json")
    if sidecar.exists():
        names = json.loads(sidecar.read_text(encoding="utf-8"))
        if isinstance(names, list) and names:
            return [str(name) for name in names]
    return list(fallback)


def predict_frame(
    model: Any,
    frame: Any,
    *,
    input_size: tuple[int, int] = (DEFAULT_INPUT_WIDTH, DEFAULT_INPUT_HEIGHT),
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict[str, Any]:
    """Classify one BGR frame and return label, score and full probabilities.

    ``input_size`` is ``(width, height)``. The model is expected to contain its own
    rescaling layer, so raw 0-255 RGB values are passed through.
    """
    import cv2
    import numpy as np

    width, height = input_size
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height))
    batch = np.expand_dims(resized, axis=0).astype(np.float32)
    probabilities = np.asarray(model(batch, training=False)).reshape(-1)
    best = int(np.argmax(probabilities))
    names = list(class_names)
    return {
        "label": names[best] if best < len(names) else str(best),
        "score": float(probabilities[best]),
        "probabilities": {
            names[index] if index < len(names) else str(index): float(value)
            for index, value in enumerate(probabilities)
        },
    }


def evaluate_classifier(
    *,
    model_path: str | Path,
    data_dir: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    height: int = DEFAULT_INPUT_HEIGHT,
    width: int = DEFAULT_INPUT_WIDTH,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Evaluate on ``data/test/<class>`` and return a full classification report."""
    import numpy as np

    from .metrics import classification_report

    test_dir = Path(data_dir) / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    model = load_classifier(model_path)
    dataset = _make_dataset(
        test_dir,
        height=height,
        width=width,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
    )
    loss, accuracy = model.evaluate(dataset, verbose=0)
    probabilities = model.predict(dataset, verbose=0)
    y_pred = np.argmax(probabilities, axis=1).tolist()
    y_true: list[int] = []
    for _, labels in dataset:
        y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())
    names = list(dataset.class_names)
    report = classification_report(
        [names[index] for index in y_true],
        [names[index] for index in y_pred],
        labels=names,
    )
    report["loss"] = float(loss)
    report["accuracy_keras"] = float(accuracy)
    return report
