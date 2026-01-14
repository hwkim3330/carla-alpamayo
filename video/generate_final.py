#!/usr/bin/env python3
"""
Final Video Generator - 맨 처음 방식 개선

문제: 모델이 21GB라 연속 추론 시 OOM
해결: 각 시나리오 결과를 미리 저장하고 영상으로 합성

이 스크립트는 두 단계로 동작:
1. 추론 실행 및 결과 저장 (pickle)
2. 저장된 결과로 영상 생성
"""

import os
import sys
import pickle
import copy
import time
import asyncio
import numpy as np
import cv2
from pathlib import Path
import torch
import gc

sys.path.insert(0, "/mnt/data/lfm_agi/alpamayo_code/src")

OUTPUT_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/output")
CACHE_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/cache")
TEMP_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/temp")

for d in [OUTPUT_DIR, CACHE_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def run_single_inference(clip_id: str, t0_us: int, output_path: str):
    """하나의 추론 실행 후 결과 저장 (별도 프로세스용)"""
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper

    print(f"Loading model...")
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16
    ).to("cuda")
    processor = helper.get_processor(model.tokenizer)

    print(f"Running inference at t0={t0_us}...")
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    frames = data["image_frames"]

    messages = helper.create_message(frames.flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    torch.cuda.synchronize()
    start = time.time()

    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=copy.deepcopy(model_inputs),
            top_p=0.98, temperature=0.6,
            num_traj_samples=1, max_generation_length=256,
            return_extra=True,
        )

    torch.cuda.synchronize()
    inference_ms = (time.time() - start) * 1000

    result = {
        "frames": frames.cpu().numpy(),
        "coc": extra["cot"][0][0][0],
        "ade": float(np.linalg.norm(
            pred_xyz.cpu()[0, 0, 0, :, :2].numpy() -
            data["ego_future_xyz"].cpu()[0, 0, :, :2].numpy(),
            axis=1
        ).mean()),
        "inference_ms": inference_ms,
        "pred": pred_xyz.cpu()[0, 0, 0].numpy(),
        "gt": data["ego_future_xyz"].cpu()[0, 0].numpy(),
        "t0_us": t0_us,
        "clip_id": clip_id,
    }

    print(f"Saving result to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"CoC: {result['coc']}")
    print(f"ADE: {result['ade']:.2f}m, Inference: {inference_ms:.0f}ms")


class VideoGenerator:
    def __init__(self):
        self.cam_labels = ["Front Left", "Front Center", "Front Right", "Rear"]

    def create_frame(self, result, t_idx=0, width=1920, height=1080):
        """프레임 생성"""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        frames = result["frames"]  # numpy array [T, C, 3, H, W]
        T, C = frames.shape[:2]
        t = t_idx % T

        # === 4개 카메라 뷰 (상단) ===
        cam_w, cam_h = 460, 259
        cam_y = 90
        cam_gap = 10
        cam_x_start = (width - 4 * cam_w - 3 * cam_gap) // 2

        for c in range(C):
            x = cam_x_start + c * (cam_w + cam_gap)
            frame = frames[t, c].transpose(1, 2, 0).astype(np.uint8)
            frame_resized = cv2.resize(frame, (cam_w, cam_h))
            canvas[cam_y:cam_y+cam_h, x:x+cam_w] = frame_resized

            # 라벨
            cv2.rectangle(canvas, (x, cam_y), (x + 110, cam_y + 22), (0, 0, 0), -1)
            cv2.putText(canvas, self.cam_labels[c], (x + 5, cam_y + 16),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if c == 1:  # Front Center 하이라이트
                cv2.rectangle(canvas, (x-2, cam_y-2), (x+cam_w+2, cam_y+cam_h+2),
                             (0, 200, 255), 2)

        # === 메인 뷰 (하단 좌측) ===
        main_frame = frames[t, 1].transpose(1, 2, 0).astype(np.uint8)
        main_w, main_h = 900, 506
        main_x, main_y = 40, 380

        main_resized = cv2.resize(main_frame, (main_w, main_h))
        canvas[main_y:main_y+main_h, main_x:main_x+main_w] = main_resized

        # 궤적 오버레이
        self._draw_trajectory(canvas, result["pred"], result["gt"],
                             main_x, main_y, main_w, main_h)

        # 범례
        cv2.circle(canvas, (main_x + 20, main_y + main_h - 35), 6, (255, 100, 0), -1)
        cv2.putText(canvas, "Predicted", (main_x + 32, main_y + main_h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1)
        cv2.circle(canvas, (main_x + 130, main_y + main_h - 35), 6, (0, 0, 255), -1)
        cv2.putText(canvas, "Ground Truth", (main_x + 142, main_y + main_h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # === 정보 패널 (하단 우측) ===
        panel_x, panel_y = 980, 380
        panel_w, panel_h = 900, 506
        cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (30, 30, 30), -1)

        # CoC
        cv2.putText(canvas, "Chain-of-Causation Reasoning", (panel_x + 20, panel_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 200, 255), 2)

        words = result["coc"].split()
        lines = []
        line = []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 45:
                lines.append(" ".join(line[:-1]))
                line = [w]
        if line:
            lines.append(" ".join(line))

        for i, ln in enumerate(lines[:4]):
            cv2.putText(canvas, ln, (panel_x + 20, panel_y + 75 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.line(canvas, (panel_x + 20, panel_y + 200), (panel_x + panel_w - 20, panel_y + 200),
                (50, 50, 50), 1)

        # 메트릭
        cv2.putText(canvas, "Performance Metrics", (panel_x + 20, panel_y + 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

        cv2.putText(canvas, f"Prediction Error (ADE): {result['ade']:.2f} meters",
                   (panel_x + 20, panel_y + 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (100, 255, 100), 2)

        cv2.putText(canvas, f"Inference Time: {result['inference_ms']:.0f} ms",
                   (panel_x + 20, panel_y + 315), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (255, 200, 100), 2)

        cv2.putText(canvas, "Trajectory: 6.4 seconds (64 waypoints @ 10Hz)",
                   (panel_x + 20, panel_y + 355), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (150, 150, 150), 1)

        cv2.line(canvas, (panel_x + 20, panel_y + 380), (panel_x + panel_w - 20, panel_y + 380),
                (50, 50, 50), 1)

        cv2.putText(canvas, "System Configuration", (panel_x + 20, panel_y + 420),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

        cv2.putText(canvas, "GPU: NVIDIA GeForce RTX 3090 (24GB)",
                   (panel_x + 20, panel_y + 455), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (150, 150, 150), 1)
        cv2.putText(canvas, "Model: Alpamayo-R1-10B (21GB)",
                   (panel_x + 20, panel_y + 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (150, 150, 150), 1)

        # === 타이틀 바 ===
        cv2.rectangle(canvas, (0, 0), (width, 70), (25, 25, 25), -1)
        cv2.putText(canvas, "NVIDIA Alpamayo-R1 | Autonomous Driving AI",
                   (40, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        cv2.circle(canvas, (width - 50, 35), 10, (0, 255, 0), -1)

        # 프레임 번호
        cv2.putText(canvas, f"Frame {t+1}/{T}", (width - 180, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        # 하단 프로그레스
        progress_y = height - 30
        progress = (t + 1) / T
        cv2.rectangle(canvas, (40, progress_y), (width - 40, progress_y + 10), (40, 40, 40), -1)
        cv2.rectangle(canvas, (40, progress_y), (int(40 + (width - 80) * progress), progress_y + 10),
                     (0, 200, 255), -1)

        return canvas

    def _draw_trajectory(self, canvas, pred, gt, x_off, y_off, w, h):
        cx, cy = x_off + w // 2, y_off + h - 40
        scale = 5

        def to_img(xyz):
            return (cx - int(xyz[1] * scale), cy - int(xyz[0] * scale))

        pts = [to_img(p) for p in gt[::4]]
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i+1], (0, 0, 255), 2)

        pts = [to_img(p) for p in pred[::4]]
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i+1], (255, 100, 0), 3)
        for pt in pts[::2]:
            cv2.circle(canvas, pt, 5, (255, 200, 0), -1)

    def create_intro(self):
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        canvas[:] = (15, 15, 15)

        cv2.putText(canvas, "NVIDIA Alpamayo-R1", (1920//2 - 320, 400),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)
        cv2.putText(canvas, "Vision-Language-Action Model", (1920//2 - 280, 500),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 2)
        cv2.putText(canvas, "for Autonomous Driving", (1920//2 - 210, 560),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 2)
        cv2.putText(canvas, "4-Camera Input | Chain-of-Causation | Trajectory Prediction",
                   (1920//2 - 420, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 255), 2)
        return canvas

    def create_outro(self):
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        canvas[:] = (15, 15, 15)

        cv2.putText(canvas, "github.com/hwkim3330/carla-alpamayo",
                   (1920//2 - 380, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (100, 200, 255), 2)
        cv2.putText(canvas, "Model: nvidia/Alpamayo-R1-10B",
                   (1920//2 - 240, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
        cv2.putText(canvas, "Subscribe for more!", (1920//2 - 160, 630),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return canvas

    def generate_video(self, results, output_path, fps=30):
        """결과들로 영상 생성"""
        width, height = 1920, 1080
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_path = str(TEMP_DIR / "temp.mp4")
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

        # 인트로
        intro = self.create_intro()
        for _ in range(fps * 3):
            out.write(cv2.cvtColor(intro, cv2.COLOR_RGB2BGR))

        # 각 결과 (애니메이션)
        frames_per_t = fps // 4
        duration_per_result = fps * 4

        for result in results:
            T = result["frames"].shape[0]
            for frame_idx in range(duration_per_result):
                t_idx = (frame_idx // frames_per_t) % T
                frame = self.create_frame(result, t_idx)
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # 아웃트로
        outro = self.create_outro()
        for _ in range(fps * 3):
            out.write(cv2.cvtColor(outro, cv2.COLOR_RGB2BGR))

        out.release()

        # H.264 변환
        os.system(f'ffmpeg -y -i {temp_path} -c:v libx264 -preset fast -crf 23 {output_path} 2>/dev/null')
        print(f"Saved: {output_path}")


def main():
    import subprocess
    import pandas as pd

    clip_ids = pd.read_parquet(
        "/mnt/data/lfm_agi/alpamayo_code/notebooks/clip_ids.parquet"
    )["clip_id"].tolist()

    # 시나리오 정의
    scenarios = [
        (clip_ids[774], 5_000_000, "construction"),
        (clip_ids[100], 5_000_000, "following"),
        (clip_ids[400], 5_000_000, "greenlight"),
    ]

    results = []

    for clip_id, t0_us, name in scenarios:
        cache_path = str(CACHE_DIR / f"{name}.pkl")

        if not os.path.exists(cache_path):
            print(f"\n=== Running inference for {name} ===")
            # 별도 프로세스로 실행
            cmd = f'''cd /mnt/data/lfm_agi/alpamayo_code && source ar1_venv/bin/activate && python -c "
import sys
sys.path.insert(0, '/mnt/data/lfm_agi/carla-alpamayo/video')
from generate_final import run_single_inference
run_single_inference('{clip_id}', {t0_us}, '{cache_path}')
"'''
            subprocess.run(cmd, shell=True, executable='/bin/bash')

        if os.path.exists(cache_path):
            print(f"Loading cached result: {cache_path}")
            with open(cache_path, 'rb') as f:
                results.append(pickle.load(f))

    if results:
        print(f"\n=== Generating video with {len(results)} results ===")
        gen = VideoGenerator()
        output_path = str(OUTPUT_DIR / "alpamayo_final.mp4")
        gen.generate_video(results, output_path)
    else:
        print("No results to generate video!")


if __name__ == "__main__":
    main()
