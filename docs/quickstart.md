# Quick Start

## Basic Usage: Asymmetric Window STFT

The core of NeuroHear is the low-latency asymmetric window STFT, which maintains good frequency resolution while achieving <10ms latency.

```python
import torch
from neurohear.core.stft import AsymmetricSTFT

# Create STFT module
# 16kHz sample rate, target latency <10ms
stft = AsymmetricSTFT(
    n_fft=256,
    hop_length=64,              # 4ms frame shift
    analysis_window_length=256,  # 16ms analysis window
    synthesis_window_length=64,  # 4ms synthesis window
)

print(f"Algorithmic latency: {stft.latency_samples} samples ({stft.latency_samples / 16000 * 1000:.1f} ms)")

# Process audio
audio = torch.randn(1, 16000)  # 1 second of audio
spectrum = stft.forward(audio)  # Analysis
reconstructed = stft.inverse(spectrum)  # Synthesis

print(f"Input: {audio.shape}")
print(f"Spectrum: {spectrum.shape}")
print(f"Reconstructed: {reconstructed.shape}")
```

## Streaming Processing

For real-time applications, use `StreamingSTFT` for frame-by-frame processing:

```python
from neurohear.core.stft import StreamingSTFT

streaming_stft = StreamingSTFT(
    n_fft=256,
    hop_length=64,
)

# Simulate real-time processing
for i in range(100):
    # Process hop_length samples at a time
    chunk = torch.randn(64)

    # Analysis
    spectrum = streaming_stft.analyze(chunk)

    # Process spectrum here (denoising, etc.)
    # processed_spectrum = your_model(spectrum)

    # Synthesis
    output = streaming_stft.synthesize(spectrum)
```

## ONNX Export

Export models to ONNX format for cross-platform deployment:

```python
from neurohear.core.stft import AsymmetricSTFT
from deploy.onnx_export import export_to_onnx, ONNXWrapper

# Create model
stft = AsymmetricSTFT(n_fft=256, hop_length=64)

# Wrap for ONNX compatibility (handles complex output)
wrapped = ONNXWrapper(stft, handle_complex=True)

# Export
export_to_onnx(wrapped, "stft.onnx", chunk_size=1024, verify=True)
```

## Next Steps

- See [API Reference](api.md) for detailed interface documentation
- Check the `examples/` directory for more examples
- Ask questions in [Discussions](https://github.com/neurohear/neurohear/discussions)
