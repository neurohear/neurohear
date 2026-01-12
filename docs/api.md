# API 参考

## neurohear.core.stft

### AsymmetricSTFT

低延迟非对称窗 STFT，用于批量处理。

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

**参数**:
- `n_fft`: FFT 大小，必须 >= analysis_window_length
- `hop_length`: 帧移（samples）
- `analysis_window_length`: 分析窗长度
- `synthesis_window_length`: 合成窗长度
- `analysis_window_type`: 分析窗类型 ("hann", "hamming", "sqrt_hann")
- `synthesis_window_type`: 合成窗类型
- `center`: 是否对输入进行填充
- `causal`: 因果模式（实时处理）

**方法**:
- `forward(x: Tensor) -> Tensor`: STFT 分析
- `inverse(X: Tensor) -> Tensor`: ISTFT 合成
- `latency_samples: int`: 算法延迟（samples）

---

### StreamingSTFT

帧级流式 STFT，用于实时处理。

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

**方法**:
- `analyze(chunk: Tensor) -> Tensor`: 分析一帧
- `synthesize(spectrum: Tensor) -> Tensor`: 合成一帧
- `reset()`: 重置内部缓冲区

---

## deploy.onnx_export

### export_to_onnx

导出 PyTorch 模型为 ONNX 格式。

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

包装模型使其 ONNX 兼容（处理复数输出等）。

```python
class ONNXWrapper(nn.Module):
    def __init__(self, model: nn.Module, handle_complex: bool = True)
```

### get_onnx_model_info

获取 ONNX 模型信息。

```python
def get_onnx_model_info(onnx_path: str | Path) -> dict
```

返回字典包含: `opset_version`, `inputs`, `outputs`, `ir_version`
