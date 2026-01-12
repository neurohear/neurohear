# Installation

## Requirements

- Python >= 3.10
- PyTorch >= 2.0

## Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/neurohear/neurohear.git
cd neurohear

# Install dependencies
uv sync

# Install development dependencies
uv sync --extra dev
```

## Using pip

```bash
# Clone the repository
git clone https://github.com/neurohear/neurohear.git
cd neurohear

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

# Install
pip install -e .
```

## Optional Dependencies

### Real-time Audio (requires portaudio)

```bash
# Ubuntu/Debian
sudo apt install portaudio19-dev

# macOS
brew install portaudio

# Install pyaudio
uv sync --extra audio
# or
pip install pyaudio
```

## Verify Installation

```python
import neurohear
print(neurohear.__version__)

from neurohear.core.stft import AsymmetricSTFT
stft = AsymmetricSTFT()
print(f"Latency: {stft.latency_samples} samples")
```
