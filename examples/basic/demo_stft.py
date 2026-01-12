"""Demo script for AsymmetricSTFT.

This script demonstrates the basic usage of the AsymmetricSTFT module,
including analysis, synthesis, and ONNX export.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from neurohear.core.stft import AsymmetricSTFT, StreamingSTFT


def demo_basic_stft():
    """Demonstrate basic STFT analysis and synthesis."""
    print("=" * 60)
    print("Demo: Basic Asymmetric STFT")
    print("=" * 60)

    # Configuration for low-latency hearing aid processing
    # 16kHz sample rate, <10ms latency target
    config = {
        "n_fft": 256,
        "hop_length": 64,  # 4ms at 16kHz
        "analysis_window_length": 256,  # 16ms - good frequency resolution
        "synthesis_window_length": 64,  # 4ms - low latency
    }

    print("\nConfiguration:")
    print("  Sample rate: 16000 Hz")
    print(f"  FFT size: {config['n_fft']}")
    print(
        f"  Hop length: {config['hop_length']} samples ({config['hop_length'] / 16000 * 1000:.1f} ms)"
    )
    print(f"  Analysis window: {config['analysis_window_length']} samples")
    print(f"  Synthesis window: {config['synthesis_window_length']} samples")

    # Create STFT module
    stft = AsymmetricSTFT(**config)
    print(
        f"\n  Algorithmic latency: {stft.latency_samples} samples ({stft.latency_samples / 16000 * 1000:.1f} ms)"
    )

    # Generate test signal: 1 second of audio
    duration = 1.0  # seconds
    sample_rate = 16000
    t = torch.linspace(0, duration, int(duration * sample_rate))

    # Mix of sine waves at different frequencies
    signal = (
        0.5 * torch.sin(2 * np.pi * 440 * t)  # A4
        + 0.3 * torch.sin(2 * np.pi * 880 * t)  # A5
        + 0.2 * torch.sin(2 * np.pi * 1320 * t)  # E6
    )
    signal = signal.unsqueeze(0)  # Add batch dimension

    print(f"\nInput signal shape: {signal.shape}")

    # Forward STFT
    spectrum = stft.forward(signal)
    print(f"Spectrum shape: {spectrum.shape}")
    print(f"  - Frequency bins: {spectrum.shape[1]}")
    print(f"  - Time frames: {spectrum.shape[2]}")

    # Inverse STFT
    reconstructed = stft.inverse(spectrum)
    print(f"Reconstructed shape: {reconstructed.shape}")

    # Compute reconstruction error
    min_len = min(signal.shape[-1], reconstructed.shape[-1])
    signal_trimmed = signal[..., :min_len]
    reconstructed_trimmed = reconstructed[..., :min_len]

    mse = torch.mean((signal_trimmed - reconstructed_trimmed) ** 2).item()
    snr = 10 * np.log10(torch.mean(signal_trimmed**2).item() / (mse + 1e-10))

    print("\nReconstruction quality:")
    print(f"  MSE: {mse:.2e}")
    print(f"  SNR: {snr:.1f} dB")


def demo_streaming_stft():
    """Demonstrate streaming (frame-by-frame) STFT processing."""
    print("\n" + "=" * 60)
    print("Demo: Streaming STFT")
    print("=" * 60)

    config = {
        "n_fft": 256,
        "hop_length": 64,
        "analysis_window_length": 256,
        "synthesis_window_length": 64,
    }

    streaming_stft = StreamingSTFT(**config)
    print(f"\nStreaming STFT latency: {streaming_stft.latency_samples} samples")

    # Generate test signal
    sample_rate = 16000
    duration = 0.1  # 100ms
    t = torch.linspace(0, duration, int(duration * sample_rate))
    signal = torch.sin(2 * np.pi * 440 * t)

    print(f"\nProcessing {len(signal)} samples in chunks of {config['hop_length']}")

    # Process frame by frame
    outputs = []
    num_frames = (len(signal) - config["n_fft"]) // config["hop_length"] + 1

    # First, fill the buffer
    for i in range(config["n_fft"] // config["hop_length"]):
        start = i * config["hop_length"]
        chunk = signal[start : start + config["hop_length"]]
        spectrum = streaming_stft.analyze(chunk)

    # Then process normally
    for i in range(num_frames):
        start = i * config["hop_length"]
        chunk = signal[start : start + config["hop_length"]]

        # Analyze
        spectrum = streaming_stft.analyze(chunk)

        # Here you could apply any frequency-domain processing
        # For this demo, we just pass through

        # Synthesize
        output = streaming_stft.synthesize(spectrum)
        outputs.append(output)

    reconstructed = torch.cat(outputs)
    print(f"Processed {len(outputs)} frames")
    print(f"Output length: {len(reconstructed)} samples")


def demo_onnx_export():
    """Demonstrate ONNX export."""
    print("\n" + "=" * 60)
    print("Demo: ONNX Export")
    print("=" * 60)

    try:
        from deploy.onnx_export import ONNXWrapper, export_to_onnx, get_onnx_model_info
    except ImportError:
        print("Skipping ONNX export demo (onnx/onnxruntime not installed)")
        return

    config = {
        "n_fft": 256,
        "hop_length": 64,
        "analysis_window_length": 256,
        "synthesis_window_length": 64,
    }

    # Create and wrap the STFT module for ONNX compatibility
    stft = AsymmetricSTFT(**config)
    wrapped_stft = ONNXWrapper(stft, handle_complex=True)

    # Export
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    onnx_path = output_dir / "asymmetric_stft.onnx"

    print(f"\nExporting to {onnx_path}")

    try:
        export_to_onnx(
            wrapped_stft,
            onnx_path,
            chunk_size=1024,
            verify=True,
            verbose=True,
        )

        # Get model info
        info = get_onnx_model_info(onnx_path)
        print("\nONNX Model Info:")
        print(f"  Opset version: {info['opset_version']}")
        print(f"  Inputs: {info['inputs']}")
        print(f"  Outputs: {info['outputs']}")

    except Exception as e:
        print(f"ONNX export failed: {e}")


def main():
    """Run all demos."""
    print("\nNeuroHear STFT Demo")
    print("===================\n")

    demo_basic_stft()
    demo_streaming_stft()
    demo_onnx_export()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
