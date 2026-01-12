"""Asymmetric window STFT for low-latency audio processing.

This module implements STFT with different analysis and synthesis window lengths,
enabling low-latency processing while maintaining good frequency resolution.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def create_window(
    window_length: int,
    window_type: Literal["hann", "hamming", "sqrt_hann", "ones"] = "hann",
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create a window function.

    Args:
        window_length: Length of the window.
        window_type: Type of window function.
        device: Device to create the window on.
        dtype: Data type of the window.

    Returns:
        Window tensor of shape (window_length,).
    """
    if window_type == "hann":
        window = torch.hann_window(
            window_length, periodic=True, device=device, dtype=dtype
        )
    elif window_type == "hamming":
        window = torch.hamming_window(
            window_length, periodic=True, device=device, dtype=dtype
        )
    elif window_type == "sqrt_hann":
        window = torch.sqrt(
            torch.hann_window(window_length, periodic=True, device=device, dtype=dtype)
        )
    elif window_type == "ones":
        window = torch.ones(window_length, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown window type: {window_type}")
    return window


class AsymmetricSTFT(nn.Module):
    """Asymmetric window STFT for low-latency processing.

    This STFT implementation uses different window lengths for analysis and synthesis,
    allowing for low-latency processing while maintaining good frequency resolution.

    For hearing aid applications with <10ms latency target at 16kHz:
    - analysis_window_length: 256 samples (16ms, good frequency resolution)
    - synthesis_window_length: 64 samples (4ms, low latency)
    - hop_length: 64 samples (4ms frame shift)

    The total algorithmic latency is approximately:
    latency = hop_length + synthesis_window_length / 2

    Args:
        n_fft: FFT size. Must be >= analysis_window_length.
        hop_length: Hop size between frames.
        analysis_window_length: Length of the analysis window.
        synthesis_window_length: Length of the synthesis window.
        analysis_window_type: Type of analysis window.
        synthesis_window_type: Type of synthesis window.
        center: Whether to pad the input on both sides.
        normalized: Whether to normalize the STFT.
        causal: If True, use causal (real-time) mode without future lookahead.

    Example:
        >>> stft = AsymmetricSTFT(
        ...     n_fft=256,
        ...     hop_length=64,
        ...     analysis_window_length=256,
        ...     synthesis_window_length=64,
        ... )
        >>> x = torch.randn(1, 16000)  # 1 second of audio at 16kHz
        >>> X = stft.forward(x)  # Analyze
        >>> y = stft.inverse(X)  # Synthesize
        >>> # y should be close to x (with some edge effects)
    """

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 64,
        analysis_window_length: int = 256,
        synthesis_window_length: int = 64,
        analysis_window_type: Literal["hann", "hamming", "sqrt_hann"] = "sqrt_hann",
        synthesis_window_type: Literal["hann", "hamming", "sqrt_hann"] = "sqrt_hann",
        center: bool = True,
        normalized: bool = False,
        causal: bool = False,
    ):
        super().__init__()

        if n_fft < analysis_window_length:
            raise ValueError(
                f"n_fft ({n_fft}) must be >= analysis_window_length ({analysis_window_length})"
            )
        if synthesis_window_length > n_fft:
            raise ValueError(
                f"synthesis_window_length ({synthesis_window_length}) must be <= n_fft ({n_fft})"
            )

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.analysis_window_length = analysis_window_length
        self.synthesis_window_length = synthesis_window_length
        self.center = center
        self.normalized = normalized
        self.causal = causal

        # Create analysis window (zero-padded to n_fft if necessary)
        analysis_window = create_window(analysis_window_length, analysis_window_type)
        if analysis_window_length < n_fft:
            # Center the window in the FFT frame
            pad_left = (n_fft - analysis_window_length) // 2
            pad_right = n_fft - analysis_window_length - pad_left
            analysis_window = F.pad(analysis_window, (pad_left, pad_right))
        self.register_buffer("analysis_window", analysis_window)

        # Create synthesis window (zero-padded to n_fft if necessary)
        synthesis_window = create_window(synthesis_window_length, synthesis_window_type)
        if synthesis_window_length < n_fft:
            # Center the window in the FFT frame
            pad_left = (n_fft - synthesis_window_length) // 2
            pad_right = n_fft - synthesis_window_length - pad_left
            synthesis_window = F.pad(synthesis_window, (pad_left, pad_right))
        self.register_buffer("synthesis_window", synthesis_window)

        # Compute normalization factor for perfect reconstruction
        # This compensates for the window overlap
        self._compute_cola_normalization()

    def _compute_cola_normalization(self) -> None:
        """Compute COLA (Constant Overlap-Add) normalization factor."""
        # Compute the overlap-add of the combined window
        combined_window = self.analysis_window * self.synthesis_window

        # Create a buffer to accumulate the overlap-add
        # We need at least n_fft + hop_length samples to see the pattern
        test_length = self.n_fft + self.hop_length * 10
        cola_sum = torch.zeros(test_length)

        for i in range(0, test_length - self.n_fft + 1, self.hop_length):
            cola_sum[i : i + self.n_fft] += combined_window.cpu()

        # Take the middle portion where the overlap is complete
        start = self.n_fft
        end = test_length - self.n_fft
        if end > start:
            # Average normalization factor (should be constant for COLA-compliant windows)
            norm_factor = cola_sum[start:end].mean()
            if norm_factor > 0:
                self.register_buffer(
                    "cola_norm", torch.tensor(1.0 / norm_factor, dtype=torch.float32)
                )
            else:
                self.register_buffer(
                    "cola_norm", torch.tensor(1.0, dtype=torch.float32)
                )
        else:
            self.register_buffer("cola_norm", torch.tensor(1.0, dtype=torch.float32))

    @property
    def latency_samples(self) -> int:
        """Return the algorithmic latency in samples.

        For causal mode, the latency is the hop_length plus half the synthesis window.
        For non-causal mode with center=True, additional latency from padding is added.
        """
        if self.causal:
            return self.hop_length + self.synthesis_window_length // 2
        elif self.center:
            return self.n_fft // 2 + self.hop_length
        else:
            return self.hop_length

    def forward(self, x: Tensor) -> Tensor:
        """Compute the STFT of the input signal.

        Args:
            x: Input tensor of shape (batch, time) or (batch, channels, time).

        Returns:
            Complex STFT tensor of shape (batch, freq_bins, frames) or
            (batch, channels, freq_bins, frames) where freq_bins = n_fft // 2 + 1.
        """
        # Handle input shape
        squeeze_channel = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, time)
            squeeze_channel = True

        batch, channels, time = x.shape

        # Pad input
        if self.center:
            if self.causal:
                # Causal padding: only pad on the left
                pad_left = self.n_fft - self.hop_length
                pad_right = 0
            else:
                # Symmetric padding
                pad_left = self.n_fft // 2
                pad_right = self.n_fft // 2
            x = F.pad(x, (pad_left, pad_right), mode="reflect")

        # Reshape for batch processing: (batch * channels, time)
        x = x.view(batch * channels, -1)

        # Compute STFT using torch.stft
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.analysis_window,
            center=False,  # We already handled padding
            return_complex=True,
            normalized=self.normalized,
        )

        # Reshape back: (batch, channels, freq_bins, frames)
        freq_bins, frames = X.shape[-2], X.shape[-1]
        X = X.view(batch, channels, freq_bins, frames)

        if squeeze_channel:
            X = X.squeeze(1)  # (batch, freq_bins, frames)

        return X

    def inverse(self, X: Tensor) -> Tensor:
        """Compute the inverse STFT.

        Args:
            X: Complex STFT tensor of shape (batch, freq_bins, frames) or
               (batch, channels, freq_bins, frames).

        Returns:
            Reconstructed time-domain signal of shape (batch, time) or
            (batch, channels, time).
        """
        # Handle input shape
        squeeze_channel = False
        if X.dim() == 3:
            X = X.unsqueeze(1)  # (batch, 1, freq_bins, frames)
            squeeze_channel = True

        batch, channels, freq_bins, frames = X.shape

        # Reshape for batch processing: (batch * channels, freq_bins, frames)
        X = X.view(batch * channels, freq_bins, frames)

        # Compute ISTFT
        x = torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.synthesis_window,
            center=False,  # We handle this ourselves
            normalized=self.normalized,
            length=None,
        )

        # Apply COLA normalization
        x = x * self.cola_norm

        # Reshape back: (batch, channels, time)
        x = x.view(batch, channels, -1)

        # Remove padding
        if self.center:
            if self.causal:
                pad_left = self.n_fft - self.hop_length
                if pad_left > 0:
                    x = x[..., pad_left:]
            else:
                pad = self.n_fft // 2
                if pad > 0:
                    x = x[..., pad:-pad] if pad < x.shape[-1] // 2 else x

        if squeeze_channel:
            x = x.squeeze(1)  # (batch, time)

        return x

    def frame(self, x: Tensor) -> Tensor:
        """Extract frames from the input signal (for streaming processing).

        This is useful for implementing frame-by-frame real-time processing.

        Args:
            x: Input tensor of shape (batch, time).

        Returns:
            Framed tensor of shape (batch, num_frames, n_fft).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch, time = x.shape

        # Pad if necessary
        if self.center:
            if self.causal:
                pad_left = self.n_fft - self.hop_length
                x = F.pad(x, (pad_left, 0), mode="reflect")
            else:
                pad = self.n_fft // 2
                x = F.pad(x, (pad, pad), mode="reflect")

        # Use unfold to extract frames
        frames = x.unfold(dimension=-1, size=self.n_fft, step=self.hop_length)

        return frames

    def extra_repr(self) -> str:
        return (
            f"n_fft={self.n_fft}, hop_length={self.hop_length}, "
            f"analysis_window_length={self.analysis_window_length}, "
            f"synthesis_window_length={self.synthesis_window_length}, "
            f"center={self.center}, causal={self.causal}, "
            f"latency_samples={self.latency_samples}"
        )


class StreamingSTFT(nn.Module):
    """Streaming STFT for real-time frame-by-frame processing.

    This class maintains internal state for processing audio in small chunks,
    suitable for real-time applications.

    Args:
        n_fft: FFT size.
        hop_length: Hop size between frames.
        analysis_window_length: Length of the analysis window.
        synthesis_window_length: Length of the synthesis window.
        analysis_window_type: Type of analysis window.
        synthesis_window_type: Type of synthesis window.

    Example:
        >>> streaming_stft = StreamingSTFT(n_fft=256, hop_length=64)
        >>> for chunk in audio_chunks:  # Each chunk is hop_length samples
        ...     spectrum = streaming_stft.analyze(chunk)
        ...     # Process spectrum...
        ...     output = streaming_stft.synthesize(spectrum)
    """

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 64,
        analysis_window_length: int = 256,
        synthesis_window_length: int = 64,
        analysis_window_type: Literal["hann", "hamming", "sqrt_hann"] = "sqrt_hann",
        synthesis_window_type: Literal["hann", "hamming", "sqrt_hann"] = "sqrt_hann",
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.analysis_window_length = analysis_window_length
        self.synthesis_window_length = synthesis_window_length

        # Create windows
        analysis_window = create_window(analysis_window_length, analysis_window_type)
        synthesis_window = create_window(synthesis_window_length, synthesis_window_type)

        # Pad windows to n_fft
        if analysis_window_length < n_fft:
            pad_left = (n_fft - analysis_window_length) // 2
            pad_right = n_fft - analysis_window_length - pad_left
            analysis_window = F.pad(analysis_window, (pad_left, pad_right))
        if synthesis_window_length < n_fft:
            pad_left = (n_fft - synthesis_window_length) // 2
            pad_right = n_fft - synthesis_window_length - pad_left
            synthesis_window = F.pad(synthesis_window, (pad_left, pad_right))

        self.register_buffer("analysis_window", analysis_window)
        self.register_buffer("synthesis_window", synthesis_window)

        # Input buffer for analysis
        self.register_buffer("input_buffer", torch.zeros(n_fft))

        # Output buffer for overlap-add synthesis
        self.register_buffer("output_buffer", torch.zeros(n_fft))

    def reset(self) -> None:
        """Reset internal buffers."""
        self.input_buffer.zero_()
        self.output_buffer.zero_()

    def analyze(self, chunk: Tensor) -> Tensor:
        """Analyze a single chunk of audio.

        Args:
            chunk: Input tensor of shape (hop_length,) or (batch, hop_length).

        Returns:
            Complex spectrum of shape (n_fft // 2 + 1,) or (batch, n_fft // 2 + 1).
        """
        squeeze_batch = False
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)
            squeeze_batch = True

        batch = chunk.shape[0]

        # Shift input buffer and add new samples
        # Note: For batch processing, we'd need per-batch buffers
        # This simple version assumes batch=1 for true streaming
        if batch == 1:
            self.input_buffer = torch.roll(self.input_buffer, -self.hop_length)
            self.input_buffer[-self.hop_length :] = chunk[0]
            windowed = self.input_buffer * self.analysis_window
            spectrum = torch.fft.rfft(windowed)
            if squeeze_batch:
                return spectrum
            return spectrum.unsqueeze(0)
        else:
            # Batch mode: process each sample independently (less efficient)
            spectra = []
            for i in range(batch):
                self.input_buffer = torch.roll(self.input_buffer, -self.hop_length)
                self.input_buffer[-self.hop_length :] = chunk[i]
                windowed = self.input_buffer * self.analysis_window
                spectra.append(torch.fft.rfft(windowed))
            return torch.stack(spectra)

    def synthesize(self, spectrum: Tensor) -> Tensor:
        """Synthesize audio from a single spectrum frame.

        Args:
            spectrum: Complex spectrum of shape (n_fft // 2 + 1,) or (batch, n_fft // 2 + 1).

        Returns:
            Output audio chunk of shape (hop_length,) or (batch, hop_length).
        """
        squeeze_batch = False
        if spectrum.dim() == 1:
            spectrum = spectrum.unsqueeze(0)
            squeeze_batch = True

        batch = spectrum.shape[0]

        if batch == 1:
            # IFFT and window
            frame = torch.fft.irfft(spectrum[0], n=self.n_fft)
            frame = frame * self.synthesis_window

            # Overlap-add
            self.output_buffer += frame

            # Extract output
            output = self.output_buffer[: self.hop_length].clone()

            # Shift output buffer
            self.output_buffer = torch.roll(self.output_buffer, -self.hop_length)
            self.output_buffer[-self.hop_length :] = 0

            if squeeze_batch:
                return output
            return output.unsqueeze(0)
        else:
            outputs = []
            for i in range(batch):
                frame = torch.fft.irfft(spectrum[i], n=self.n_fft)
                frame = frame * self.synthesis_window
                self.output_buffer += frame
                output = self.output_buffer[: self.hop_length].clone()
                self.output_buffer = torch.roll(self.output_buffer, -self.hop_length)
                self.output_buffer[-self.hop_length :] = 0
                outputs.append(output)
            return torch.stack(outputs)

    @property
    def latency_samples(self) -> int:
        """Return the algorithmic latency in samples."""
        return self.n_fft - self.hop_length + self.hop_length

    def extra_repr(self) -> str:
        return (
            f"n_fft={self.n_fft}, hop_length={self.hop_length}, "
            f"latency_samples={self.latency_samples}"
        )
