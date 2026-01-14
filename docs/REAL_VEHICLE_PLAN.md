# Real Vehicle Test Plan

## Overview

Alpamayo-R1 모델을 실제 차량에 적용하기 위한 단계별 계획입니다.

## Phase 1: Simulation Validation

### 목표
CARLA 시뮬레이터에서 end-to-end 검증

### Tasks
- [ ] CARLA-Alpamayo 통합 테스트
- [ ] 다양한 시나리오 시뮬레이션
  - 도심 주행
  - 고속도로 주행
  - 교차로 통과
  - 장애물 회피
- [ ] 실시간 추론 성능 측정
- [ ] Closed-loop 제어 검증

### Success Criteria
- 추론 지연: < 100ms
- Collision-free rate: > 99%
- Route completion: > 95%

## Phase 2: Hardware Setup

### Camera Configuration

Alpamayo는 멀티카메라 입력을 사용합니다:

```
┌─────────────────────────────────────┐
│           Front Center              │
│              ┌───┐                  │
│              │ C1│                  │
│              └───┘                  │
│  ┌───┐                    ┌───┐    │
│  │C2 │   [Vehicle]        │C3 │    │
│  └───┘                    └───┘    │
│ Front Left             Front Right  │
│                                     │
│         ┌───┐     ┌───┐            │
│         │C4 │     │C5 │            │
│         └───┘     └───┘            │
│        Side L    Side R            │
│                                     │
│              ┌───┐                  │
│              │C6 │                  │
│              └───┘                  │
│             Rear                    │
└─────────────────────────────────────┘
```

### Recommended Hardware

| Component | Specification |
|-----------|---------------|
| Compute | NVIDIA Jetson AGX Orin (64GB) |
| Camera | GMSL2 cameras × 6-8 |
| GPS/IMU | Novatel PwrPak7 or similar |
| CAN Interface | Kvaser or Peak CAN |
| Storage | NVMe SSD 1TB+ |

### Software Stack

```
┌─────────────────────────────────────┐
│          Application Layer          │
│  ┌─────────────────────────────┐   │
│  │    Alpamayo-R1 Inference    │   │
│  └─────────────────────────────┘   │
├─────────────────────────────────────┤
│          Middleware Layer           │
│  ┌──────────┐  ┌──────────────┐    │
│  │   ROS2   │  │  DriveWorks  │    │
│  └──────────┘  └──────────────┘    │
├─────────────────────────────────────┤
│           Driver Layer              │
│  ┌──────┐ ┌─────┐ ┌──────────┐    │
│  │Camera│ │ CAN │ │ GPS/IMU  │    │
│  └──────┘ └─────┘ └──────────┘    │
└─────────────────────────────────────┘
```

## Phase 3: Safety Framework

### Safety Layers

1. **Model Output Validation**
   - Trajectory smoothness check
   - Physical feasibility check
   - Out-of-distribution detection

2. **Fallback Controller**
   - Rule-based backup
   - Emergency stop capability
   - Geofencing

3. **Human Override**
   - Driver monitoring
   - Manual takeover interface
   - Remote intervention

### Safety Metrics

| Metric | Target |
|--------|--------|
| False positive rate | < 1% |
| Intervention rate | < 0.1/km |
| Mean time to takeover | > 5 sec |

## Phase 4: Data Collection

### On-Road Data

실차 주행 중 수집할 데이터:

```python
{
    "timestamp": "2026-01-14T10:30:00",
    "cameras": {
        "front_center": "image_fc.jpg",
        "front_left": "image_fl.jpg",
        ...
    },
    "vehicle_state": {
        "position": [x, y, z],
        "velocity": [vx, vy, vz],
        "acceleration": [ax, ay, az],
        "steering_angle": 0.05,
        "throttle": 0.3,
        "brake": 0.0
    },
    "model_output": {
        "coc": "Maintain lane and speed",
        "trajectory": [[x1,y1], [x2,y2], ...],
        "inference_time_ms": 45
    },
    "ground_truth": {
        "actual_trajectory": [[x1,y1], ...]
    }
}
```

### Test Routes

| Route | Type | Distance | Duration |
|-------|------|----------|----------|
| Route A | Urban | 10 km | 30 min |
| Route B | Highway | 30 km | 20 min |
| Route C | Mixed | 20 km | 40 min |

## Phase 5: Evaluation

### Metrics

1. **Prediction Accuracy**
   - ADE (Average Displacement Error)
   - FDE (Final Displacement Error)
   - Miss Rate

2. **Driving Quality**
   - Smoothness (jerk)
   - Comfort (lateral acceleration)
   - Efficiency (time, distance)

3. **Safety**
   - Near-miss events
   - Traffic violations
   - Intervention frequency

### Benchmark Comparison

| Model | ADE | FDE | Inference |
|-------|-----|-----|-----------|
| Alpamayo-R1 | TBD | TBD | TBD |
| Baseline | - | - | - |

## Timeline

```
Phase 1 (Simulation)     ████████░░░░░░░░
Phase 2 (Hardware)       ░░░░████████░░░░
Phase 3 (Safety)         ░░░░░░░░████████
Phase 4 (Data Collection)░░░░░░░░░░░░████
Phase 5 (Evaluation)     ░░░░░░░░░░░░░░██
```

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Model latency too high | High | Edge optimization, TensorRT |
| Camera calibration drift | Medium | Auto-calibration routine |
| Weather conditions | Medium | Domain adaptation |
| Regulatory approval | High | Start with private roads |

## Contact

- Project Lead: [TBD]
- Safety Officer: [TBD]
- Vehicle Integration: [TBD]

## References

- [NVIDIA DRIVE Documentation](https://developer.nvidia.com/drive)
- [Alpamayo Paper](https://arxiv.org/abs/2511.00088)
- [SAE J3016 Automation Levels](https://www.sae.org/standards/content/j3016_202104/)
