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

## Project Structure

```
neurohear/
├── src/
│   ├── neurohear/          # Core toolkit
│   │   ├── core/           # STFT, audio I/O, pipeline
│   │   ├── models/         # Denoising, feedback, compensation
│   │   └── utils/          # Utilities
│   └── deploy/             # Deployment tools
│       ├── onnx_export.py  # ONNX export
│       └── platforms/      # Platform-specific code
├── examples/               # Usage examples
└── tests/                  # Unit tests
```

## Algorithms

- **Denoising**: Deep learning based noise reduction
- **Feedback Suppression**: Acoustic feedback cancellation
- **Hearing Compensation**: Audiogram-based frequency gain

## License

MIT License
