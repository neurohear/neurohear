# NeuroHear

A deep learning toolkit for hearing aid algorithms.

## Why NeuroHear?

Our ultimate goal is to **lower the barrier to hearing aid algorithm development**.

Existing open-source hearing aid platforms like **OpenMHA** and **Tympan** are primarily built on traditional digital signal processing (DSP) algorithms. While effective, they lack native support for deep learning approaches that have shown superior performance in speech/audio-related tasks.

**NeuroHear** aims to fill this gap by providing:

- **Native deep learning support**: All algorithms implemented in PyTorch with ONNX export
- **Low-latency design**: Target latency <10ms, suitable for real-time hearing aid applications
- **End-to-end neural solutions**: Denoising, feedback suppression, and hearing compensation all powered by neural networks
- **Easy deployment**: ONNX export enables cross-platform deployment (Windows, Raspberry Pi, etc.)

### Comparison

| Feature | OpenMHA | Tympan | NeuroHear |
|---------|---------|--------|-----------|
| Core Framework | C++ DSP | Arduino/C++ | PyTorch |
| Deep Learning | Limited | Limited | Native |
| ONNX Export | No | No | Yes |
| Target Latency | <10ms | <10ms | <10ms |
| Algorithms | Traditional DSP | Traditional DSP | Neural Network |

## Roadmap

- [x] Asymmetric window STFT (low-latency)
- [x] ONNX export utilities
- [ ] Real-time audio I/O
- [ ] Neural encoder-decoder for low-latency processing
- [ ] Neural network based denoising
- [ ] Neural network based feedback suppression
- [ ] Neural network based hearing compensation
- [ ] Raspberry Pi deployment example
- [ ] Windows deployment example
- [ ] Hearing assessment tools
- [ ] 3D-printable hardware designs

## Features

- **Asymmetric Window STFT**: Low-latency frequency transform with different analysis/synthesis windows
- **Neural Denoising**: Deep learning based noise reduction (coming soon)
- **Neural Feedback Suppression**: Acoustic feedback cancellation using neural networks (coming soon)
- **Neural Hearing Compensation**: Audiogram-based frequency compensation powered by neural networks (coming soon)

## Documentation

- [Installation](docs/installation.md)
- [Quick Start](docs/quickstart.md)
- [API Reference](docs/api.md)
- [Contributing](docs/contributing.md)

## Community

- [Discussions](https://github.com/neurohear/neurohear/discussions) - Questions & ideas
- [Issues](https://github.com/neurohear/neurohear/issues) - Bug reports & feature requests

## License

MIT License
