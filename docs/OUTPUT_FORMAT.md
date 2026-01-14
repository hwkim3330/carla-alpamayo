# Alpamayo-R1 Output Format Specification

## Overview

Alpamayo-R1은 Vision-Language-Action (VLA) 모델로, 멀티카메라 입력을 받아 **Chain-of-Causation 추론**과 **미래 경로 예측**을 출력합니다.

## Input Format

### Multi-Camera Images
```
Shape: [batch, num_cameras, channels, height, width]
Example: [1, 8, 3, 480, 640]

카메라 배치:
- Camera 1-4: 전방/측면 뷰 (시간 t-3, t-2, t-1, t)
- Camera 5-8: 추가 뷰 (rear, left, right 등)
```

### Ego Vehicle History
```python
ego_history_xyz: torch.Tensor
    Shape: [batch, 1, history_len, 3]
    Description: 과거 자차 위치 (x, y, z) in meters
    Coordinate: Ego vehicle frame

ego_history_rot: torch.Tensor
    Shape: [batch, 1, history_len, 3, 3]
    Description: 과거 자차 회전 행렬 (rotation matrix)
```

## Output Format

### 1. Chain-of-Causation (CoC) Reasoning

자연어로 된 주행 의사결정 추론입니다.

```python
extra["cot"]: List[List[List[str]]]
    Shape: [batch_size, num_traj_sets, num_traj_samples]

Example:
[
    [
        ["Nudge to the left to increase clearance from the construction cones encroaching into the lane"]
    ]
]
```

#### CoC 패턴 예시

| 상황 | CoC 출력 |
|------|----------|
| 공사 구간 | "Nudge to the left to increase clearance from the construction cones" |
| 앞차 감속 | "Slow down to maintain a safe following distance from the decelerating lead car" |
| 녹색 신호 | "Accelerate to proceed through the intersection since the traffic light turns green" |
| 주차 차량 | "Nudge left to increase clearance from the parked vehicle on the right shoulder" |
| 보행자 | "Slow down and prepare to stop for pedestrians crossing" |

### 2. Trajectory Prediction

미래 6.4초 동안의 자차 경로를 예측합니다.

```python
pred_xyz: torch.Tensor
    Shape: [batch, num_traj_sets, num_traj_samples, num_waypoints, 3]
    Example: [1, 1, 1, 64, 3]

    - num_waypoints: 64 (10Hz × 6.4초)
    - 좌표: (x, y, z) in meters, ego vehicle frame
    - x: 전방 (+) / 후방 (-)
    - y: 좌측 (+) / 우측 (-)
    - z: 상방 (+) / 하방 (-)

pred_rot: torch.Tensor
    Shape: [batch, num_traj_sets, num_traj_samples, num_waypoints, 3, 3]
    Description: 각 waypoint에서의 회전 행렬
```

#### Trajectory 시각화

```
    ↑ Y (좌측)
    |
    |    * * * * (predicted trajectory)
    |  *
    | *
    |*
    +----------→ X (전방)
   Ego
```

### 3. JSON Output Structure

배치 추론 시 저장되는 JSON 형식:

```json
{
  "clip_id": "030c760c-ae38-49aa-9ad8-f5650a545d26",
  "chain_of_causation": "Nudge to the left to increase clearance...",
  "metrics": {
    "minADE": 0.7522,
    "maxError": 2.5598
  },
  "trajectory": {
    "num_waypoints": 64,
    "duration_sec": 6.4,
    "predicted_xyz": [
      [0.5, 0.1, 0.0],
      [1.2, 0.15, 0.0],
      ...
    ],
    "predicted_rotation": [
      [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      ...
    ]
  },
  "timestamp": "2026-01-13T17:35:00"
}
```

## Evaluation Metrics

### Average Displacement Error (ADE)

예측 경로와 실제 경로 간의 평균 거리 오차:

```python
ADE = mean(||pred_xy - gt_xy||)  # meters
```

| 성능 수준 | ADE 범위 |
|----------|----------|
| Excellent | < 0.5m |
| Good | 0.5 - 1.0m |
| Acceptable | 1.0 - 2.0m |
| Poor | > 2.0m |

### Final Displacement Error (FDE)

마지막 waypoint에서의 오차:

```python
FDE = ||pred_xy[-1] - gt_xy[-1]||  # meters
```

## Usage Example

```python
import torch
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

# Load model
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")

# Inference
with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,
        max_generation_length=256,
        return_extra=True,
    )

# Extract outputs
reasoning = extra["cot"][0][0][0]  # Chain-of-Causation string
trajectory = pred_xyz[0, 0, 0]     # [64, 3] waypoints
rotation = pred_rot[0, 0, 0]       # [64, 3, 3] rotation matrices
```

## Notes

- **비결정적 출력**: VLA 모델은 sampling으로 인해 매번 다른 결과 생성 가능
- **GPU 메모리**: `num_traj_samples` 증가 시 메모리 사용량 증가
- **좌표계**: Ego vehicle frame 기준 (차량 중심이 원점)
