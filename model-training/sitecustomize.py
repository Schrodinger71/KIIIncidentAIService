"""Local startup patch for Keras save compatibility during fine-tuning.

This module is auto-imported by Python's ``site`` initialization when the
current working directory is ``model-training``. We use it to strip the
unsupported ``options`` kwarg only for native ``.keras`` saves in older
``tf.keras`` combinations that still pass it from ``ModelCheckpoint``.
"""

from __future__ import annotations

from os import PathLike
from typing import Any


def _is_native_keras_path(filepath: Any) -> bool:
    if isinstance(filepath, PathLike):
        filepath = filepath.__fspath__()
    if isinstance(filepath, bytes):
        filepath = filepath.decode()
    return isinstance(filepath, str) and filepath.lower().endswith(".keras")


def _patch_keras_save_model() -> None:
    try:
        import keras
        from keras.src.saving import saving_api
    except Exception:
        return

    original_save_model = saving_api.save_model

    if getattr(original_save_model, "_kii_options_patch", False):
        return

    def patched_save_model(
        model: Any,
        filepath: Any,
        overwrite: bool = True,
        save_format: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if _is_native_keras_path(filepath):
            kwargs.pop("options", None)
        return original_save_model(
            model,
            filepath,
            overwrite=overwrite,
            save_format=save_format,
            **kwargs,
        )

    patched_save_model._kii_options_patch = True  # type: ignore[attr-defined]

    saving_api.save_model = patched_save_model

    if hasattr(keras, "saving") and hasattr(keras.saving, "save_model"):
        keras.saving.save_model = patched_save_model
    if hasattr(keras, "models") and hasattr(keras.models, "save_model"):
        keras.models.save_model = patched_save_model

    try:
        import tensorflow as tf
    except Exception:
        return

    if hasattr(tf.keras, "saving") and hasattr(tf.keras.saving, "save_model"):
        tf.keras.saving.save_model = patched_save_model
    if hasattr(tf.keras, "models") and hasattr(tf.keras.models, "save_model"):
        tf.keras.models.save_model = patched_save_model


_patch_keras_save_model()
