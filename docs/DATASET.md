# PhysicalAI-Autonomous-Vehicles Dataset

## Overview

NVIDIA에서 공개한 자율주행 데이터셋으로, Alpamayo 모델 학습 및 평가에 사용됩니다.

## Source

- **HuggingFace**: [nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- **License**: Research use only (비상업적 연구 목적)
- **Access**: Gated dataset (HuggingFace 승인 필요)

## Dataset Statistics

| 항목 | 값 |
|------|-----|
| Total Clips | 1,181 |
| Training Data | 80,000+ hours driving |
| Images | 1+ Billion |
| CoC Reasoning Traces | 700K |

## Data Format

### Clip Structure

각 클립은 고유 UUID로 식별됩니다:
```
clip_id: "030c760c-ae38-49aa-9ad8-f5650a545d26"
```

### Contents per Clip

```python
data = load_physical_aiavdataset(clip_id)

data.keys():
- "image_frames"      # 멀티카메라 이미지 [num_cams, C, H, W]
- "ego_history_xyz"   # 과거 자차 위치
- "ego_history_rot"   # 과거 자차 회전
- "ego_future_xyz"    # 미래 자차 위치 (GT)
- "ego_future_rot"    # 미래 자차 회전 (GT)
```

### Image Frames

```
Cameras: 8개 (전방, 측면, 후방 등)
Resolution: Variable (typically 480x640 or higher)
Format: RGB, uint8
```

## Sample Scenarios

데이터셋에 포함된 다양한 주행 상황:

### 1. 공사 구간 (Construction Zone)
- 공사 콘, 바리케이드
- 차선 변경 필요 상황

### 2. 도심 주행 (Urban Driving)
- 신호등, 정지 신호
- 교차로 통과

### 3. 차량 상호작용 (Vehicle Interaction)
- 앞차 따라가기
- 주차 차량 회피
- 끼어들기 대응

### 4. 보행자/자전거 (VRU)
- 횡단보도
- 자전거 도로

## Loading Data

### Basic Usage

```python
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

# Load specific clip
clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
data = load_physical_aiavdataset(clip_id)

# Load with specific timestamp
data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
```

### List All Clips

```python
import pandas as pd

clip_ids = pd.read_parquet("notebooks/clip_ids.parquet")["clip_id"].tolist()
print(f"Available clips: {len(clip_ids)}")  # 1181
```

## Tested Clips

| Clip Index | Clip ID | Scenario |
|------------|---------|----------|
| 0 | 0347d9f9-... | 주차 차량 회피 |
| 100 | 0fc8cf3b-... | 앞차 감속 대응 |
| 200 | a382ab2b-... | 선행 차량 추종 |
| 400 | 74d763f1-... | 녹색 신호 직진 |
| 774 | 030c760c-... | 공사 콘 회피 |

## Cache Location

다운로드된 데이터는 HuggingFace 캐시에 저장됩니다:

```
~/.cache/huggingface/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/
```

## Video Data

현재 데이터셋은 **개별 이미지 프레임**으로 제공됩니다.
- 연속 비디오 파일은 제공되지 않음
- 이미지 시퀀스를 직접 비디오로 변환 가능

```python
import imageio

# Create video from frames
frames = data["image_frames"].permute(0, 2, 3, 1).numpy()
imageio.mimwrite("output.mp4", frames, fps=10)
```

## Related Resources

- [Alpamayo Paper (arXiv)](https://arxiv.org/abs/2511.00088)
- [NVIDIA Alpamayo Blog](https://developer.nvidia.com/drive/alpamayo)
- [Model Weights](https://huggingface.co/nvidia/Alpamayo-R1-10B)
