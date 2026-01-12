# 安装指南

## 环境要求

- Python >= 3.10
- PyTorch >= 2.0

## 使用 uv 安装（推荐）

```bash
# 克隆仓库
git clone https://github.com/neurohear/neurohear.git
cd neurohear

# 安装依赖
uv sync

# 安装开发依赖
uv sync --extra dev
```

## 使用 pip 安装

```bash
# 克隆仓库
git clone https://github.com/neurohear/neurohear.git
cd neurohear

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows

# 安装
pip install -e .
```

## 可选依赖

### 实时音频（需要 portaudio）

```bash
# Ubuntu/Debian
sudo apt install portaudio19-dev

# macOS
brew install portaudio

# 安装 pyaudio
uv sync --extra audio
# 或
pip install pyaudio
```

## 验证安装

```python
import neurohear
print(neurohear.__version__)

from neurohear.core.stft import AsymmetricSTFT
stft = AsymmetricSTFT()
print(f"Latency: {stft.latency_samples} samples")
```
