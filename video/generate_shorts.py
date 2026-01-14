#!/usr/bin/env python3
"""
Shorts Video Generator (9:16 Portrait)
Uses cached results from generate_final.py
"""

import os
import pickle
import numpy as np
import cv2
from pathlib import Path

CACHE_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/cache")
OUTPUT_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/output")
TEMP_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/temp")


class ShortsGenerator:
    def __init__(self):
        self.cam_labels = ["Front Left", "Front Center", "Front Right", "Rear"]
        self.width = 1080
        self.height = 1920

    def create_frame(self, result, t_idx=0):
        """세로 프레임 생성"""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (15, 15, 15)

        frames = result["frames"]
        T, C = frames.shape[:2]
        t = t_idx % T

        # === 타이틀 ===
        cv2.putText(canvas, "NVIDIA Alpamayo-R1", (self.width//2 - 200, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(canvas, "Autonomous Driving AI", (self.width//2 - 170, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

        # === 메인 뷰 (Front Center) ===
        main_frame = frames[t, 1].transpose(1, 2, 0).astype(np.uint8)
        main_w, main_h = 1000, 563
        main_x = (self.width - main_w) // 2
        main_y = 130

        main_resized = cv2.resize(main_frame, (main_w, main_h))
        canvas[main_y:main_y+main_h, main_x:main_x+main_w] = main_resized

        # 궤적 오버레이
        self._draw_trajectory(canvas, result["pred"], result["gt"],
                             main_x, main_y, main_w, main_h)

        # 범례
        cv2.circle(canvas, (main_x + 20, main_y + main_h - 30), 5, (255, 100, 0), -1)
        cv2.putText(canvas, "Predicted", (main_x + 30, main_y + main_h - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)
        cv2.circle(canvas, (main_x + 120, main_y + main_h - 30), 5, (0, 0, 255), -1)
        cv2.putText(canvas, "Ground Truth", (main_x + 130, main_y + main_h - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # === 4개 카메라 (2x2 그리드) ===
        cam_w, cam_h = 480, 270
        cam_y_start = 720
        cam_gap = 10

        for c in range(4):
            row, col = c // 2, c % 2
            x = (self.width - 2*cam_w - cam_gap) // 2 + col * (cam_w + cam_gap)
            y = cam_y_start + row * (cam_h + cam_gap)

            frame = frames[t, c].transpose(1, 2, 0).astype(np.uint8)
            frame_resized = cv2.resize(frame, (cam_w, cam_h))
            canvas[y:y+cam_h, x:x+cam_w] = frame_resized

            cv2.rectangle(canvas, (x, y), (x + 100, y + 20), (0, 0, 0), -1)
            cv2.putText(canvas, self.cam_labels[c], (x + 5, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            if c == 1:
                cv2.rectangle(canvas, (x-2, y-2), (x+cam_w+2, y+cam_h+2),
                             (0, 200, 255), 2)

        # === CoC 추론 ===
        coc_y = 1310
        cv2.putText(canvas, "Chain-of-Causation:", (50, coc_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

        words = result["coc"].split()
        lines = []
        line = []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 38:
                lines.append(" ".join(line[:-1]))
                line = [w]
        if line:
            lines.append(" ".join(line))

        for i, ln in enumerate(lines[:3]):
            cv2.putText(canvas, ln, (50, coc_y + 35 + i * 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # === 메트릭 ===
        metric_y = 1470
        cv2.line(canvas, (50, metric_y - 15), (self.width - 50, metric_y - 15),
                (40, 40, 40), 1)

        cv2.putText(canvas, f"Prediction Error: {result['ade']:.2f}m",
                   (50, metric_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (100, 255, 100), 2)

        cv2.putText(canvas, f"Inference: {result['inference_ms']:.0f}ms",
                   (self.width - 280, metric_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (255, 200, 100), 2)

        # === 하단 정보 ===
        cv2.putText(canvas, "Model: Alpamayo-R1-10B | GPU: RTX 3090",
                   (self.width//2 - 230, 1550), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (100, 100, 100), 1)

        # 프레임 번호
        cv2.putText(canvas, f"Frame {t+1}/{T}", (self.width - 130, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # 프로그레스 바
        progress_y = self.height - 40
        progress = (t + 1) / T
        cv2.rectangle(canvas, (50, progress_y), (self.width - 50, progress_y + 8),
                     (40, 40, 40), -1)
        cv2.rectangle(canvas, (50, progress_y),
                     (int(50 + (self.width - 100) * progress), progress_y + 8),
                     (0, 200, 255), -1)

        # GitHub
        cv2.putText(canvas, "github.com/hwkim3330/carla-alpamayo",
                   (self.width//2 - 200, self.height - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

        return canvas

    def _draw_trajectory(self, canvas, pred, gt, x_off, y_off, w, h):
        cx, cy = x_off + w // 2, y_off + h - 50
        scale = 4

        def to_img(xyz):
            return (cx - int(xyz[1] * scale), cy - int(xyz[0] * scale))

        pts = [to_img(p) for p in gt[::4]]
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i+1], (0, 0, 255), 2)

        pts = [to_img(p) for p in pred[::4]]
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i+1], (255, 100, 0), 2)
        for pt in pts[::2]:
            cv2.circle(canvas, pt, 4, (255, 200, 0), -1)

    def create_intro(self):
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (15, 15, 15)

        cv2.putText(canvas, "NVIDIA", (self.width//2 - 100, self.height//2 - 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (100, 200, 100), 3)
        cv2.putText(canvas, "Alpamayo-R1", (self.width//2 - 180, self.height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
        cv2.putText(canvas, "Vision-Language-Action Model",
                   (self.width//2 - 220, self.height//2 + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
        cv2.putText(canvas, "for Autonomous Driving",
                   (self.width//2 - 160, self.height//2 + 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
        return canvas

    def create_outro(self):
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (15, 15, 15)

        cv2.putText(canvas, "Subscribe!", (self.width//2 - 100, self.height//2 - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(canvas, "github.com/hwkim3330", (self.width//2 - 150, self.height//2 + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
        cv2.putText(canvas, "/carla-alpamayo", (self.width//2 - 110, self.height//2 + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
        return canvas

    def generate_video(self, results, output_path, fps=30):
        """세로 영상 생성"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_path = str(TEMP_DIR / "temp_shorts.mp4")
        out = cv2.VideoWriter(temp_path, fourcc, fps, (self.width, self.height))

        # 인트로
        intro = self.create_intro()
        for _ in range(fps * 2):
            out.write(cv2.cvtColor(intro, cv2.COLOR_RGB2BGR))

        # 시나리오
        frames_per_t = fps // 4
        duration_per_result = fps * 3

        for result in results:
            T = result["frames"].shape[0]
            for frame_idx in range(duration_per_result):
                t_idx = (frame_idx // frames_per_t) % T
                frame = self.create_frame(result, t_idx)
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # 아웃트로
        outro = self.create_outro()
        for _ in range(fps * 2):
            out.write(cv2.cvtColor(outro, cv2.COLOR_RGB2BGR))

        out.release()

        os.system(f'ffmpeg -y -i {temp_path} -c:v libx264 -preset fast -crf 23 {output_path} 2>/dev/null')
        print(f"Saved shorts: {output_path}")


def main():
    results = []
    for name in ["construction", "following", "greenlight"]:
        cache_path = CACHE_DIR / f"{name}.pkl"
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                results.append(pickle.load(f))
            print(f"Loaded: {name}")

    if results:
        gen = ShortsGenerator()
        output_path = str(OUTPUT_DIR / "alpamayo_final_shorts.mp4")
        gen.generate_video(results, output_path)
    else:
        print("No cached results found!")


if __name__ == "__main__":
    main()
