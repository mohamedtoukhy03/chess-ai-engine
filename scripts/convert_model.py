#!/usr/bin/env python3
"""
Convert a trained Keras model to ONNX for C++ inference.

Usage:
  python convert_model.py \
      --keras "chess_se_resnet_best.keras" \
      --onnx "../models/chess_eval.onnx"

Dependencies:
  pip install tensorflow tf2onnx onnx
"""

from __future__ import annotations

import argparse
import pathlib
import tensorflow as tf
import tf2onnx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Keras -> ONNX converter")
    p.add_argument("--keras", required=True, help="Path to .keras model")
    p.add_argument("--onnx", required=True, help="Output ONNX path")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    keras_path = pathlib.Path(args.keras)
    onnx_path = pathlib.Path(args.onnx)

    if not keras_path.exists():
        raise FileNotFoundError(f"Keras model not found: {keras_path}")

    print(f"Loading Keras model from: {keras_path}")
    model = tf.keras.models.load_model(keras_path, compile=False)

    # Explicit signatures to preserve dual-input model API:
    # board: (None, 8, 8, 12), meta: (None, 8)
    input_signature = (
        tf.TensorSpec((None, 8, 8, 12), tf.float32, name="board"),
        tf.TensorSpec((None, 8), tf.float32, name="meta"),
    )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting ONNX to: {onnx_path} (opset={args.opset})")
    _model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=args.opset,
        output_path=str(onnx_path),
    )
    print("Done.")


if __name__ == "__main__":
    main()
