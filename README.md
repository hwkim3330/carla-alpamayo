# CARLA-Alpamayo

NVIDIA Alpamayo VLA (Vision-Language-Action) 모델을 CARLA 시뮬레이터와 통합한 자율주행 에이전트입니다.

## Overview

[Alpamayo-R1](https://github.com/NVlabs/alpamayo)은 NVIDIA가 개발한 자율주행을 위한 VLA 모델로, Chain-of-Thought (CoT) 추론 기능을 통해 복잡한 주행 상황에서 단계별 판단을 수행합니다.

이 프로젝트는 Alpamayo 모델을 [CARLA](https://carla.org/) 자율주행 시뮬레이터에서 실행할 수 있도록 브릿지 코드를 제공합니다.

## Features

- CARLA 시뮬레이터와 Alpamayo VLA 모델 통합
- 멀티 센서 지원 (카메라, GNSS, IMU, LiDAR)
- Chain-of-Thought 추론 기반 주행 결정
- 녹화 및 분석 도구 포함
- GPU 없이 테스트 가능한 더미 모드

## Requirements

### Hardware
- NVIDIA GPU (VRAM 24GB 이상 권장)
- 16GB+ RAM

### Software
- Ubuntu 20.04 / 22.04
- Python 3.8+
- CARLA 0.9.13+
- CUDA 11.8+

## Installation

### 1. CARLA 설치

```bash
# CARLA 다운로드 (0.9.15 권장)
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
mkdir -p ~/carla && tar -xvf CARLA_0.9.15.tar.gz -C ~/carla

# CARLA Python API 설치
pip install carla
```

### 2. 프로젝트 설치

```bash
git clone https://github.com/hwkim3330/carla-alpamayo.git
cd carla-alpamayo

# 의존성 설치
pip install -r requirements.txt

# PyTorch 설치 (CUDA 버전에 맞게)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Alpamayo 모델 다운로드

```bash
# HuggingFace 로그인
huggingface-cli login

# 모델 다운로드 (약 22GB)
python scripts/download_model.py
```

**Note:** 모델 접근 권한이 필요합니다:
- [Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B) 접근 요청

## Quick Start

### 1. CARLA 서버 시작

```bash
# 터미널 1: CARLA 시작
./scripts/start_carla.sh ~/carla

# 또는 직접 실행
cd ~/carla && ./CarlaUE4.sh -prefernvidia
```

### 2. 에이전트 실행

```bash
# 터미널 2: Alpamayo 에이전트 실행
cd examples

# 더미 모드 (GPU 없이 테스트)
python run_agent.py --dummy --frames 500

# 전체 모델 사용
python run_agent.py --frames 1000 --command "follow the road"
```

### 3. 녹화 모드

```bash
# 주행 데이터 녹화
python run_with_recording.py --dummy --frames 300 --output ../recordings
```

## Usage

### Python API

```python
from src.carla_alpamayo_agent import CarlaAlpamayoAgent, AgentConfig

# 설정
config = AgentConfig(
    host="localhost",
    port=2000,
    use_dummy_model=False,  # True for testing without GPU
)

# Context manager 사용
with CarlaAlpamayoAgent(config) as agent:
    agent.run(
        max_frames=1000,
        navigation_command="turn left at the intersection",
    )
```

### Navigation Commands

Alpamayo는 자연어 명령을 이해합니다:

- `"follow the road"` - 차선 유지
- `"turn left at the intersection"` - 교차로에서 좌회전
- `"turn right"` - 우회전
- `"stop"` - 정지
- `"change lane to the left"` - 좌측 차선 변경

## Project Structure

```
carla-alpamayo/
├── src/
│   ├── __init__.py
│   ├── carla_alpamayo_agent.py  # 메인 에이전트 클래스
│   ├── alpamayo_wrapper.py      # Alpamayo 모델 래퍼
│   └── sensor_manager.py        # CARLA 센서 관리
├── examples/
│   ├── run_agent.py             # 기본 실행 예제
│   └── run_with_recording.py    # 녹화 예제
├── scripts/
│   ├── start_carla.sh           # CARLA 시작 스크립트
│   └── download_model.py        # 모델 다운로드
├── configs/
│   └── default_config.yaml      # 기본 설정
├── requirements.txt
├── setup.py
└── README.md
```

## Configuration

`configs/default_config.yaml`에서 설정을 수정할 수 있습니다:

```yaml
carla:
  host: localhost
  port: 2000

model:
  name: "nvidia/Alpamayo-R1-10B"
  device: "cuda"

control:
  target_fps: 10.0
  max_speed_kmh: 30.0
```

## Troubleshooting

### CARLA 연결 실패
```bash
# CARLA가 실행 중인지 확인
ps aux | grep CarlaUE4

# 포트 확인
netstat -tlnp | grep 2000
```

### GPU 메모리 부족
```bash
# 더미 모드로 테스트
python run_agent.py --dummy

# 또는 더 작은 모델 사용 (추후 지원)
```

### 모델 로드 실패
```bash
# HuggingFace 로그인 확인
huggingface-cli whoami

# 캐시 삭제 후 재다운로드
rm -rf ~/.cache/huggingface/hub/models--nvidia--Alpamayo*
python scripts/download_model.py
```

## References

- [NVIDIA Alpamayo](https://github.com/NVlabs/alpamayo)
- [AlpaSim Simulator](https://github.com/NVlabs/alpasim)
- [CARLA Simulator](https://carla.org/)
- [Alpamayo Technical Blog](https://developer.nvidia.com/blog/building-autonomous-vehicles-that-reason-with-nvidia-alpamayo/)

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first.
