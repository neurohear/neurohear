"""ONNX export utilities for NeuroHear models.

This module provides tools to export PyTorch models to ONNX format
for cross-platform deployment.
"""

import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    sample_rate: int = 16000,
    chunk_size: int = 64,
    opset_version: int = 17,
    dynamic_axes: dict | None = None,
    verify: bool = True,
    verbose: bool = False,
) -> Path:
    """Export a PyTorch model to ONNX format.

    Args:
        model: The PyTorch model to export.
        output_path: Path to save the ONNX model.
        sample_rate: Sample rate of the audio (for documentation).
        chunk_size: Expected input chunk size in samples.
        opset_version: ONNX opset version.
        dynamic_axes: Dynamic axes specification for variable-length inputs.
        verify: Whether to verify the exported model.
        verbose: Whether to print export details.

    Returns:
        Path to the exported ONNX model.

    Example:
        >>> from neurohear.core.stft import AsymmetricSTFT
        >>> stft = AsymmetricSTFT(n_fft=256, hop_length=64)
        >>> export_to_onnx(stft, "stft.onnx", chunk_size=64)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, chunk_size)

    # Default dynamic axes for audio processing
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size", 1: "time"},
            "output": {0: "batch_size"},
        }

    # Export
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            verbose=verbose,
        )

    if verbose:
        print(f"Model exported to {output_path}")

    # Verify the exported model
    if verify:
        _verify_onnx_model(output_path, model, dummy_input, verbose)

    return output_path


def _verify_onnx_model(
    onnx_path: Path,
    pytorch_model: nn.Module,
    dummy_input: Tensor,
    verbose: bool = False,
) -> None:
    """Verify that the ONNX model produces the same output as PyTorch.

    Args:
        onnx_path: Path to the ONNX model.
        pytorch_model: The original PyTorch model.
        dummy_input: Input tensor for verification.
        verbose: Whether to print verification details.
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as e:
        warnings.warn(f"Cannot verify ONNX model: {e}")
        return

    # Check ONNX model validity
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    if verbose:
        print("ONNX model structure is valid")

    # Run inference with ONNX Runtime
    ort_session = ort.InferenceSession(str(onnx_path))
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    # Run inference with PyTorch
    with torch.no_grad():
        pytorch_outputs = pytorch_model(dummy_input)

    # Handle complex outputs
    if pytorch_outputs.is_complex():
        # ONNX doesn't support complex numbers directly
        # Compare real and imaginary parts separately if needed
        pytorch_outputs_np = torch.view_as_real(pytorch_outputs).numpy()
    else:
        pytorch_outputs_np = pytorch_outputs.numpy()

    # Compare outputs
    try:
        import numpy as np

        # Allow for some numerical tolerance
        if np.allclose(ort_outputs[0], pytorch_outputs_np, rtol=1e-3, atol=1e-5):
            if verbose:
                print("ONNX model output matches PyTorch output")
        else:
            max_diff = np.max(np.abs(ort_outputs[0] - pytorch_outputs_np))
            warnings.warn(
                f"ONNX model output differs from PyTorch. Max difference: {max_diff}"
            )
    except Exception as e:
        warnings.warn(f"Could not compare outputs: {e}")


class ONNXWrapper(nn.Module):
    """Wrapper to make models ONNX-compatible.

    Some PyTorch operations are not directly supported by ONNX.
    This wrapper provides alternatives for common operations.

    Args:
        model: The model to wrap.
        handle_complex: Whether to convert complex outputs to real.
    """

    def __init__(self, model: nn.Module, handle_complex: bool = True):
        super().__init__()
        self.model = model
        self.handle_complex = handle_complex

    def forward(self, x: Tensor) -> Tensor:
        output = self.model(x)

        if self.handle_complex and output.is_complex():
            # Convert complex to real by stacking real and imaginary parts
            # Shape: (..., 2) where last dim is [real, imag]
            output = torch.view_as_real(output)

        return output


def export_streaming_model(
    model: nn.Module,
    output_path: str | Path,
    n_fft: int = 256,
    hop_length: int = 64,
    opset_version: int = 17,
    verify: bool = True,
    verbose: bool = False,
) -> Path:
    """Export a streaming audio processing model to ONNX.

    This function is specifically designed for frame-by-frame processing models
    used in real-time audio applications.

    Args:
        model: The streaming model to export.
        output_path: Path to save the ONNX model.
        n_fft: FFT size (for creating dummy input).
        hop_length: Hop length (chunk size for streaming).
        opset_version: ONNX opset version.
        verify: Whether to verify the exported model.
        verbose: Whether to print export details.

    Returns:
        Path to the exported ONNX model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # For streaming models, input is typically a single frame
    dummy_input = torch.randn(1, hop_length)

    # Dynamic axes for streaming
    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            verbose=verbose,
        )

    if verbose:
        print(f"Streaming model exported to {output_path}")

    if verify:
        _verify_onnx_model(output_path, model, dummy_input, verbose)

    return output_path


def get_onnx_model_info(onnx_path: str | Path) -> dict:
    """Get information about an ONNX model.

    Args:
        onnx_path: Path to the ONNX model.

    Returns:
        Dictionary containing model information.
    """
    import onnx

    model = onnx.load(str(onnx_path))

    # Get input info
    inputs = []
    for inp in model.graph.input:
        shape = [
            dim.dim_value or dim.dim_param for dim in inp.type.tensor_type.shape.dim
        ]
        inputs.append(
            {
                "name": inp.name,
                "shape": shape,
                "dtype": onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type),
            }
        )

    # Get output info
    outputs = []
    for out in model.graph.output:
        shape = [
            dim.dim_value or dim.dim_param for dim in out.type.tensor_type.shape.dim
        ]
        outputs.append(
            {
                "name": out.name,
                "shape": shape,
                "dtype": onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type),
            }
        )

    return {
        "opset_version": model.opset_import[0].version,
        "inputs": inputs,
        "outputs": outputs,
        "ir_version": model.ir_version,
    }
