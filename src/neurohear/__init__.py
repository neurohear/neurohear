"""NeuroHear - A deep learning toolkit for hearing aid algorithms.

Key features:
- Low-latency STFT with asymmetric windows (<10ms latency target)
- ONNX export for cross-platform deployment
- Real-time audio processing pipeline
"""

__version__ = "0.1.0"

from neurohear.core.stft import AsymmetricSTFT, StreamingSTFT

__all__ = [
    "__version__",
    "AsymmetricSTFT",
    "StreamingSTFT",
]
