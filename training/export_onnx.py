"""Export TensorFlow Keras checkpoint to ONNX with board/meta inputs."""

from __future__ import annotations

import os

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import tf2onnx

from config import CHECKPOINT_PATH, ONNX_EXPORT_PATH


def export_to_onnx() -> None:
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Missing trained checkpoint: {CHECKPOINT_PATH}. Run training first."
        )

    model = tf.keras.models.load_model(CHECKPOINT_PATH, compile=False, safe_mode=False)
    os.makedirs(os.path.dirname(ONNX_EXPORT_PATH), exist_ok=True)

    spec = (
        tf.TensorSpec((None, 8, 8, 12), tf.float32, name="board"),
        tf.TensorSpec((None, 8), tf.float32, name="meta"),
    )
    tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=17, output_path=ONNX_EXPORT_PATH
    )
    print(f"ONNX model exported to: {ONNX_EXPORT_PATH}")

    onnx_model = onnx.load(ONNX_EXPORT_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX checker: PASSED")

    sess = ort.InferenceSession(ONNX_EXPORT_PATH)
    test_board = np.random.randn(1, 8, 8, 12).astype(np.float32)
    test_meta = np.random.randn(1, 8).astype(np.float32)
    onnx_out = sess.run(None, {"board": test_board, "meta": test_meta})[0]
    keras_out = model.predict([test_board, test_meta], verbose=0)
    diff = float(np.max(np.abs(onnx_out - keras_out)))
    print(f"Max abs diff (Keras vs ONNX): {diff:.8f}")


if __name__ == "__main__":
    export_to_onnx()
