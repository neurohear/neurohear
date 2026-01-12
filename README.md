# NeuroHear

A deep learning toolkit for hearing aid algorithms.

## Features

- Native PyTorch implementation with ONNX export support
- Low-latency design (<10ms target)
- Modular architecture for easy algorithm development
- Cross-platform deployment (Windows, Raspberry Pi)

## Installation

```bash
# Using uv (recommended)
git clone https://github.com/neurohear/neurohear.git
cd neurohear
uv sync

# Or using pip
pip install -e .
```

## Quick Start

```python
from neurohear.core.stft import AsymmetricSTFT

# Create STFT with asymmetric windows for low latency
stft = AsymmetricSTFT(
    n_fft=256,
    hop_length=64,
    analysis_window_length=256,
    synthesis_window_length=64,
)

# Process audio
spectrum = stft.forward(audio)
reconstructed = stft.inverse(spectrum)
```

## Documentation

- [Installation](docs/installation.md) - Setup guide
- [Quick Start](docs/quickstart.md) - Get started with examples
- [API Reference](docs/api.md) - Detailed API documentation
- [Contributing](docs/contributing.md) - How to contribute

## Roadmap

- [x] Asymmetric window STFT (low-latency)
- [x] ONNX export utilities
- [ ] Real-time audio I/O (pyaudio)
- [ ] Denoising model (lightweight, <10ms)
- [ ] Feedback suppression
- [ ] Hearing compensation (audiogram-based)
- [ ] Raspberry Pi deployment example
- [ ] Windows deployment example

## Community

- [Discussions](https://github.com/neurohear/neurohear/discussions) - Questions & ideas
- [Issues](https://github.com/neurohear/neurohear/issues) - Bug reports & feature requests

## License

MIT License
