# Alpamayo-R1-10B Inference Test Results

## Test Environment

| Item | Value |
|------|-------|
| Date | 2026-01-13 |
| GPU | NVIDIA GeForce RTX 3090 (24GB) |
| PyTorch | 2.8.0 |
| Transformers | 4.57.1 |
| Model | nvidia/Alpamayo-R1-10B |
| Model Size | 21GB (10B parameters) |

## Dataset

- **Source**: `nvidia/PhysicalAI-Autonomous-Vehicles` (HuggingFace)
- **Clip ID**: `030c760c-ae38-49aa-9ad8-f5650a545d26`
- **Timestamp**: `t0_us=5_100_000`
- **Cache Size**: ~16MB (example clip)

## Inference Results

### Chain-of-Causation (CoC) Reasoning

```
"Nudge to the left to increase clearance from the construction cones encroaching into the lane"
```

모델이 차선에 침범한 공사 콘을 인식하고, 왼쪽으로 살짝 이동하여 거리를 확보하라는 추론을 생성함.

### Trajectory Prediction

| Metric | Value |
|--------|-------|
| minADE | 0.75218153 meters |
| Trajectory Length | 64 waypoints (6.4 seconds @ 10Hz) |
| Output Format | (x, y, z) position + rotation matrix |

## How to Reproduce

### 1. Setup Environment

```bash
cd /mnt/data/lfm_agi/alpamayo_code
source ar1_venv/bin/activate
uv sync --active
```

### 2. Run Inference Test

```bash
python src/alpamayo_r1/test_inference.py
```

### 3. Expected Output

```
Loading dataset for clip_id: 030c760c-ae38-49aa-9ad8-f5650a545d26...
Dataset loaded.
Fetching 5 files: 100%|██████████| 5/5 [04:12<00:00, 50.42s/it]
Loading checkpoint shards: 100%|██████████| 5/5 [00:00<00:00, 42.57it/s]
Chain-of-Causation (per trajectory):
 [['Nudge to the left to increase clearance from the construction cones encroaching into the lane']]
minADE: 0.75218153 meters
```

## Model Architecture

Alpamayo-R1은 Vision-Language-Action (VLA) 모델로:

1. **Input**: Multi-camera driving video (front, left, right 등)
2. **Backbone**: Cosmos-Reason VLM (10B parameters)
3. **Output**:
   - Chain-of-Causation reasoning trace
   - 6.4초 미래 경로 예측 (64 waypoints)

## Notes

- VLA 모델은 trajectory sampling으로 인해 비결정적(nondeterministic) 출력 생성
- `num_traj_samples=1`로 설정 시 GPU 메모리 호환성 확보 (24GB에서 동작)
- 더 많은 trajectory 샘플링은 `num_traj_samples` 값 증가로 가능

## Directory Structure

```
/mnt/data/lfm_agi/
├── alpamayo_code/           # NVlabs/alpamayo 공식 코드
│   └── ar1_venv/            # Python 가상환경
├── models/
│   └── alpamayo-r1-10b/     # 모델 웨이트 (21GB)
├── carla/                   # CARLA 0.9.16
└── carla-alpamayo/          # CARLA-Alpamayo 통합 프로젝트
```
