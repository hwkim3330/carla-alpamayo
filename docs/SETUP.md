# Installation & Setup Guide

## System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 24GB VRAM (RTX 3090) | 48GB+ (A6000, H100) |
| RAM | 32GB | 64GB |
| Storage | 100GB | 200GB+ |
| CPU | 8 cores | 16+ cores |

### Software

| Software | Version |
|----------|---------|
| Ubuntu | 20.04 / 22.04 |
| Python | 3.10+ |
| CUDA | 12.1+ |
| PyTorch | 2.0+ |

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/hwkim3330/carla-alpamayo.git
cd carla-alpamayo
```

### 2. Install Alpamayo

```bash
# Clone official Alpamayo code
git clone https://github.com/NVlabs/alpamayo.git alpamayo_code
cd alpamayo_code

# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create virtual environment
uv venv ar1_venv
source ar1_venv/bin/activate
uv sync --active
```

### 3. Download Model Weights

```bash
# Option 1: Using Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/Alpamayo-R1-10B', local_dir='models/alpamayo-r1-10b')
"

# Option 2: Using huggingface-cli
huggingface-cli download nvidia/Alpamayo-R1-10B --local-dir models/alpamayo-r1-10b
```

Model size: **~22GB**

### 4. HuggingFace Authentication

데이터셋 접근을 위해 HuggingFace 인증 필요:

```bash
# Request access at:
# https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles

# Login
huggingface-cli login
# or
python -c "from huggingface_hub import login; login()"
```

### 5. Install CARLA (Optional)

CARLA 시뮬레이터 연동 시:

```bash
# Download CARLA 0.9.15+
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.15.tar.gz
tar -xzf CARLA_0.9.15.tar.gz -C /opt/carla

# Install CARLA Python API
pip install carla==0.9.15
```

## Directory Structure

설치 완료 후 디렉토리 구조:

```
/mnt/data/lfm_agi/
├── alpamayo_code/           # NVlabs/alpamayo 공식 코드
│   ├── ar1_venv/            # Python 가상환경
│   ├── src/alpamayo_r1/     # 모델 소스코드
│   └── notebooks/           # 예제 노트북
├── models/
│   └── alpamayo-r1-10b/     # 모델 웨이트 (21GB)
├── carla/                   # CARLA 0.9.16
│   ├── CarlaUE4.sh          # CARLA 서버 실행
│   └── PythonAPI/           # Python 클라이언트
└── carla-alpamayo/          # 이 프로젝트
    ├── docs/                # 문서
    ├── src/                 # 소스코드
    └── examples/            # 예제
```

## Quick Test

### Test Alpamayo Inference

```bash
cd alpamayo_code
source ar1_venv/bin/activate

# Run test script
python src/alpamayo_r1/test_inference.py
```

Expected output:
```
Chain-of-Causation: "Nudge to the left to increase clearance..."
minADE: 0.75 meters
```

### Test CARLA Connection

```bash
# Terminal 1: Start CARLA server
cd /opt/carla
./CarlaUE4.sh -RenderOffScreen

# Terminal 2: Test connection
python -c "
import carla
client = carla.Client('localhost', 2000)
print(f'CARLA version: {client.get_server_version()}')
"
```

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce trajectory samples
num_traj_samples=1  # Instead of 3 or more

# Use lower precision
model = model.to(torch.bfloat16)
```

### Flash Attention Issues

```python
# Use alternative attention
config.attn_implementation = "sdpa"
```

### HuggingFace Access Denied

1. https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles 에서 접근 요청
2. 승인 후 `huggingface-cli login` 재실행

## Next Steps

1. [OUTPUT_FORMAT.md](OUTPUT_FORMAT.md) - 출력 형식 이해
2. [DATASET.md](DATASET.md) - 데이터셋 정보
3. [INFERENCE_TEST.md](INFERENCE_TEST.md) - 추론 테스트 결과
4. [REAL_VEHICLE_PLAN.md](REAL_VEHICLE_PLAN.md) - 실차 테스트 계획
