# Alpamayo Demo Video Plan

## Video Concept

**제목**: "Alpamayo-R1: AI가 운전 상황을 이해하고 판단하는 방법"

**길이**: 3-5분

**목표**: Alpamayo VLA 모델이 다양한 주행 상황에서 어떻게 추론하고 경로를 예측하는지 시각적으로 보여주기

---

## Video Structure

```
┌────────────────────────────────────────────────────────────┐
│  0:00 - 0:15  │  INTRO                                     │
│               │  - 타이틀 카드                              │
│               │  - "NVIDIA Alpamayo-R1 Demo"               │
├────────────────────────────────────────────────────────────┤
│  0:15 - 0:45  │  WHAT IS ALPAMAYO?                         │
│               │  - 모델 설명 텍스트                         │
│               │  - 입력(카메라) → 출력(추론+경로) 다이어그램 │
├────────────────────────────────────────────────────────────┤
│  0:45 - 3:30  │  SCENARIO DEMOS (각 30초 x 6개)            │
│               │                                            │
│               │  [Scenario 1] 공사 구간                    │
│               │  [Scenario 2] 앞차 감속                    │
│               │  [Scenario 3] 녹색 신호 직진               │
│               │  [Scenario 4] 주차 차량 회피               │
│               │  [Scenario 5] 보행자/자전거                │
│               │  [Scenario 6] 고속도로 합류                │
├────────────────────────────────────────────────────────────┤
│  3:30 - 4:00  │  RESULTS SUMMARY                           │
│               │  - 성능 지표 (ADE, 추론 시간)              │
│               │  - 다양한 CoC 패턴 모음                    │
├────────────────────────────────────────────────────────────┤
│  4:00 - 4:15  │  OUTRO                                     │
│               │  - GitHub 링크                             │
│               │  - "More scenarios coming soon"            │
└────────────────────────────────────────────────────────────┘
```

---

## Per-Scenario Layout

각 시나리오 30초 구성:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                     │   │
│  │              MAIN VIEW (Front Camera)               │   │
│  │                   with trajectory overlay           │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ Cam Left │ │ Cam Right│ │ Cam Rear │ │   BEV    │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  🤖 Chain-of-Causation:                             │   │
│  │  "Nudge left to avoid construction cones..."        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ADE: 0.75m  |  Inference: 45ms  |  Scenario: Construction │
└─────────────────────────────────────────────────────────────┘
```

---

## Technical Implementation

### 1. Frame Extraction
```python
# 각 클립에서 연속 프레임 추출 (2-3초 분량)
frames = data["image_frames"]  # [time, cameras, C, H, W]
```

### 2. Trajectory Overlay
```python
# 예측 경로를 이미지에 오버레이
# - 빨간색: Ground Truth
# - 파란색: Prediction
# - 반투명 원으로 waypoints 표시
```

### 3. Text Overlay
```python
# CoC 추론 텍스트를 하단에 표시
# - 타자기 효과로 한 글자씩 나타나기
# - 또는 fade-in 효과
```

### 4. Video Composition
```python
# FFmpeg 또는 moviepy로 조합
# - 배경음악 추가
# - 전환 효과 (fade, slide)
```

---

## Clip Selection Criteria

1181개 클립 중 다양한 시나리오 선별:

| Category | Target Count | Selection Criteria |
|----------|--------------|-------------------|
| Construction | 2 | 콘, 바리케이드 visible |
| Following | 2 | 앞차와 상호작용 |
| Intersection | 2 | 신호등 visible |
| Lane Change | 2 | 측면 이동 trajectory |
| Pedestrian | 2 | VRU 존재 |
| Highway | 2 | 고속 주행 |

총 12개 클립 → 6개 시나리오로 편집

---

## Output Specifications

| Property | Value |
|----------|-------|
| Resolution | 1920x1080 (1080p) |
| Frame Rate | 30 fps |
| Format | MP4 (H.264) |
| Audio | Background music (royalty-free) |
| Duration | 3-5 minutes |

---

## File Structure

```
video/
├── VIDEO_PLAN.md           # 이 문서
├── generate_video.py       # 영상 생성 스크립트
├── assets/
│   ├── intro.png           # 인트로 이미지
│   ├── outro.png           # 아웃트로 이미지
│   ├── font.ttf            # 텍스트용 폰트
│   └── music.mp3           # 배경음악
├── clips/
│   ├── scenario_01/        # 각 시나리오별 프레임
│   ├── scenario_02/
│   └── ...
└── output/
    └── alpamayo_demo.mp4   # 최종 영상
```

---

## Next Steps

1. [ ] 다양한 시나리오 클립 탐색 및 선별
2. [ ] 영상 생성 스크립트 작성
3. [ ] 인트로/아웃트로 이미지 제작
4. [ ] 배경음악 선정
5. [ ] 영상 렌더링
6. [ ] 유튜브 업로드
