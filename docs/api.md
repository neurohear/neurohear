# API Reference

## neurohear.core.stft

### AsymmetricSTFT

Low-latency asymmetric window STFT for batch processing.

```python
class AsymmetricSTFT(nn.Module):
    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 64,
        analysis_window_length: int = 256,
        synthesis_window_length: int = 64,
        analysis_window_type: str = "sqrt_hann",
        synthesis_window_type: str = "sqrt_hann",
        center: bool = True,
        normalized: bool = False,
        causal: bool = False,
    )
```

**Parameters**:
- `n_fft`: FFT size, must be >= analysis_window_length
- `hop_length`: Frame shift (samples)
- `analysis_window_length`: Analysis window length
- `synthesis_window_length`: Synthesis window length
- `analysis_window_type`: Analysis window type ("hann", "hamming", "sqrt_hann")
- `synthesis_window_type`: Synthesis window type
- `center`: Whether to pad the input
- `causal`: Causal mode (for real-time processing)

**Methods**:
- `forward(x: Tensor) -> Tensor`: STFT analysis
- `inverse(X: Tensor) -> Tensor`: ISTFT synthesis
- `latency_samples: int`: Algorithmic latency (samples)

---

### StreamingSTFT

Frame-level streaming STFT for real-time processing.

```python
class StreamingSTFT(nn.Module):
    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 64,
        analysis_window_length: int = 256,
        synthesis_window_length: int = 64,
        analysis_window_type: str = "sqrt_hann",
        synthesis_window_type: str = "sqrt_hann",
    )
```

**Methods**:
- `analyze(chunk: Tensor) -> Tensor`: Analyze one frame
- `synthesize(spectrum: Tensor) -> Tensor`: Synthesize one frame
- `reset()`: Reset internal buffers

---

## deploy.onnx_export

### export_to_onnx

Export a PyTorch model to ONNX format.

```python
def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    sample_rate: int = 16000,
    chunk_size: int = 64,
    opset_version: int = 17,
    dynamic_axes: dict | None = None,
    verify: bool = True,
    verbose: bool = False,
) -> Path
```

### ONNXWrapper

Wrapper to make models ONNX-compatible (handles complex outputs, etc.).

```python
class ONNXWrapper(nn.Module):
    def __init__(self, model: nn.Module, handle_complex: bool = True)
```

### get_onnx_model_info

Get information about an ONNX model.

```python
def get_onnx_model_info(onnx_path: str | Path) -> dict
```

Returns a dictionary containing: `opset_version`, `inputs`, `outputs`, `ir_version`
