# 快速开始

## 基本用法：非对称窗 STFT

NeuroHear 的核心是低延迟的非对称窗 STFT，可以在保持良好频率分辨率的同时实现 <10ms 延迟。

```python
import torch
from neurohear.core.stft import AsymmetricSTFT

# 创建 STFT 模块
# 16kHz 采样率，目标延迟 <10ms
stft = AsymmetricSTFT(
    n_fft=256,
    hop_length=64,              # 4ms 帧移
    analysis_window_length=256,  # 16ms 分析窗
    synthesis_window_length=64,  # 4ms 合成窗
)

print(f"算法延迟: {stft.latency_samples} samples ({stft.latency_samples / 16000 * 1000:.1f} ms)")

# 处理音频
audio = torch.randn(1, 16000)  # 1秒音频
spectrum = stft.forward(audio)  # 分析
reconstructed = stft.inverse(spectrum)  # 合成

print(f"输入: {audio.shape}")
print(f"频谱: {spectrum.shape}")
print(f"重构: {reconstructed.shape}")
```

## 流式处理

对于实时应用，使用 `StreamingSTFT` 进行帧级处理：

```python
from neurohear.core.stft import StreamingSTFT

streaming_stft = StreamingSTFT(
    n_fft=256,
    hop_length=64,
)

# 模拟实时处理
for i in range(100):
    # 每次处理 hop_length 个样本
    chunk = torch.randn(64)

    # 分析
    spectrum = streaming_stft.analyze(chunk)

    # 这里可以对频谱进行处理（降噪等）
    # processed_spectrum = your_model(spectrum)

    # 合成
    output = streaming_stft.synthesize(spectrum)
```

## ONNX 导出

将模型导出为 ONNX 格式，方便跨平台部署：

```python
from neurohear.core.stft import AsymmetricSTFT
from deploy.onnx_export import export_to_onnx, ONNXWrapper

# 创建模型
stft = AsymmetricSTFT(n_fft=256, hop_length=64)

# 包装为 ONNX 兼容格式（处理复数输出）
wrapped = ONNXWrapper(stft, handle_complex=True)

# 导出
export_to_onnx(wrapped, "stft.onnx", chunk_size=1024, verify=True)
```

## 下一步

- 查看 [API 参考](api.md) 了解详细接口
- 查看 `examples/` 目录获取更多示例
- 在 [Discussions](https://github.com/neurohear/neurohear/discussions) 提问交流
