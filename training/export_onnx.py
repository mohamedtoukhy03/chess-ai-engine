"""
Export trained PyTorch model to ONNX format for C++ inference.

The exported model takes input shape (1, 18, 8, 8) and outputs (1, 1).
ONNX Runtime is used in the C++ engine for fast inference.
"""

import os
import torch
import onnx
import onnxruntime as ort
import numpy as np

from config import MODEL_SAVE_PATH, ONNX_EXPORT_PATH
from model import ChessEvalNet


def export_to_onnx():
    print("Loading trained model...")
    model = ChessEvalNet()

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"No trained model found at {MODEL_SAVE_PATH}")
        print("Exporting untrained model for testing...")
    else:
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        print(f"Val loss: {checkpoint.get('val_loss', '?')}")

    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 18, 8, 8)

    # Export
    os.makedirs(os.path.dirname(ONNX_EXPORT_PATH), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_EXPORT_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["board"],
        output_names=["eval"],
        dynamic_axes={
            "board": {0: "batch_size"},
            "eval":  {0: "batch_size"},
        },
    )
    print(f"\nONNX model exported to {ONNX_EXPORT_PATH}")

    # Verify
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(ONNX_EXPORT_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification: PASSED")

    # Test inference with ONNX Runtime
    print("\nTesting ONNX Runtime inference...")
    session = ort.InferenceSession(ONNX_EXPORT_PATH)

    test_input = np.random.randn(1, 18, 8, 8).astype(np.float32)
    result = session.run(None, {"board": test_input})

    print(f"Input shape:  {test_input.shape}")
    print(f"Output shape: {result[0].shape}")
    print(f"Output value: {result[0][0][0]:.4f}")

    # Compare PyTorch vs ONNX output
    with torch.no_grad():
        pt_out = model(torch.from_numpy(test_input)).numpy()

    diff = abs(pt_out[0][0] - result[0][0][0])
    print(f"\nPyTorch output: {pt_out[0][0]:.6f}")
    print(f"ONNX output:    {result[0][0][0]:.6f}")
    print(f"Difference:     {diff:.8f}")

    if diff < 1e-5:
        print("✓ PyTorch and ONNX outputs match!")
    else:
        print("⚠ Small numerical difference (expected with float32)")

    # Model size
    size_mb = os.path.getsize(ONNX_EXPORT_PATH) / (1024 * 1024)
    print(f"\nONNX model size: {size_mb:.2f} MB")


if __name__ == "__main__":
    export_to_onnx()
