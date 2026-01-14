#!/usr/bin/env python3
"""
Alpamayo Demo Video Generator v2

Features:
- 16:9 landscape (YouTube)
- 9:16 portrait (Shorts/Reels)
- 4-camera split view
- Inference time display
- System environment info
"""

import os
import sys
import copy
import time
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch

sys.path.insert(0, "/mnt/data/lfm_agi/alpamayo_code/src")

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

OUTPUT_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ScenarioResult:
    clip_id: str
    scenario_name: str
    scenario_name_kr: str
    coc: str
    ade: float
    inference_time_ms: float
    frames: np.ndarray
    trajectory_pred: np.ndarray
    trajectory_gt: np.ndarray


class VideoGeneratorV2:
    def __init__(self):
        self.model = None
        self.processor = None

        # System info
        self.gpu_name = "NVIDIA RTX 3090"
        self.gpu_memory = "24GB"
        self.model_name = "Alpamayo-R1-10B"
        self.model_size = "21GB"

        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                   capture_output=True, text=True)
            self.gpu_name = result.stdout.strip()
        except:
            pass

    def load_model(self):
        if self.model is None:
            print("Loading model...")
            self.model = AlpamayoR1.from_pretrained(
                "nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16
            ).to("cuda")
            self.processor = helper.get_processor(self.model.tokenizer)

    def run_inference(self, clip_id: str, scenario_name: str, scenario_name_kr: str) -> Optional[ScenarioResult]:
        self.load_model()

        try:
            print(f"Processing: {scenario_name} ({clip_id[:8]}...)")
            data = load_physical_aiavdataset(clip_id)
            frames = data["image_frames"].flatten(0, 1).permute(0, 2, 3, 1).numpy()

            messages = helper.create_message(data["image_frames"].flatten(0, 1))
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
                continue_final_message=True, return_dict=True, return_tensors="pt",
            )
            model_inputs = {
                "tokenized_data": inputs,
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
            }
            model_inputs = helper.to_device(model_inputs, "cuda")

            # Measure inference time
            torch.cuda.synchronize()
            start_time = time.time()

            torch.cuda.manual_seed_all(42)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                    data=copy.deepcopy(model_inputs),
                    top_p=0.98, temperature=0.6,
                    num_traj_samples=1, max_generation_length=256,
                    return_extra=True,
                )

            torch.cuda.synchronize()
            inference_time = (time.time() - start_time) * 1000

            coc = extra["cot"][0][0][0]
            gt_xy = data["ego_future_xyz"].cpu()[0, 0].numpy()
            pred_xy = pred_xyz.cpu()[0, 0, 0].numpy()
            ade = np.linalg.norm(pred_xy[:, :2] - gt_xy[:, :2], axis=1).mean()

            print(f"  CoC: {coc[:50]}...")
            print(f"  ADE: {ade:.2f}m, Inference: {inference_time:.0f}ms")

            return ScenarioResult(
                clip_id=clip_id,
                scenario_name=scenario_name,
                scenario_name_kr=scenario_name_kr,
                coc=coc,
                ade=ade,
                inference_time_ms=inference_time,
                frames=frames,
                trajectory_pred=pred_xy,
                trajectory_gt=gt_xy
            )
        except Exception as e:
            print(f"Error: {e}")
            return None

    def create_landscape_frame(self, result: ScenarioResult, width=1920, height=1080) -> np.ndarray:
        """16:9 landscape frame with 4-camera split"""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (25, 25, 25)

        # 4-camera grid (2x2) - left side
        cam_w, cam_h = 440, 248
        cam_x_start, cam_y_start = 40, 120
        gap = 10

        cam_indices = [0, 1, 4, 5]  # Front cameras + side cameras
        cam_labels = ["Front Left", "Front Center", "Side Left", "Side Right"]

        for i, (cam_idx, label) in enumerate(zip(cam_indices, cam_labels)):
            row, col = i // 2, i % 2
            x = cam_x_start + col * (cam_w + gap)
            y = cam_y_start + row * (cam_h + gap)

            if cam_idx < len(result.frames):
                frame = cv2.resize(result.frames[cam_idx], (cam_w, cam_h))
                canvas[y:y+cam_h, x:x+cam_w] = frame

            # Camera label
            cv2.putText(canvas, label, (x + 5, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Main view with trajectory - right side
        main_x, main_y = 960, 120
        main_w, main_h = 920, 500

        if len(result.frames) > 1:
            main_frame = cv2.resize(result.frames[1], (main_w, main_h))  # Front center
            canvas[main_y:main_y+main_h, main_x:main_x+main_w] = main_frame

        # Draw trajectory overlay
        self._draw_trajectory(canvas, result, main_x, main_y, main_w, main_h)

        # Title bar
        cv2.rectangle(canvas, (0, 0), (width, 80), (35, 35, 35), -1)
        cv2.putText(canvas, "NVIDIA Alpamayo-R1 Demo", (40, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Scenario name (Korean)
        cv2.putText(canvas, result.scenario_name_kr, (width - 300, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 255), 2)

        # CoC box - bottom left
        coc_y = 640
        cv2.rectangle(canvas, (40, coc_y), (900, coc_y + 100), (40, 40, 40), -1)
        cv2.putText(canvas, "Chain-of-Causation:", (60, coc_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        coc_text = result.coc[:80] + "..." if len(result.coc) > 80 else result.coc
        cv2.putText(canvas, f'"{coc_text}"', (60, coc_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Metrics panel - bottom right
        metrics_y = 640
        cv2.rectangle(canvas, (960, metrics_y), (width - 40, metrics_y + 180), (40, 40, 40), -1)

        # Performance metrics
        cv2.putText(canvas, "Performance", (980, metrics_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        cv2.putText(canvas, f"ADE: {result.ade:.2f} m", (980, metrics_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        cv2.putText(canvas, f"Inference: {result.inference_time_ms:.0f} ms", (980, metrics_y + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)

        # System info
        cv2.putText(canvas, "System", (1200, metrics_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        cv2.putText(canvas, f"GPU: {self.gpu_name}", (1200, metrics_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(canvas, f"Model: {self.model_size}", (1200, metrics_y + 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Trajectory info
        cv2.putText(canvas, "Trajectory: 6.4s (64 waypoints @ 10Hz)", (980, metrics_y + 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(canvas, f"Clip: {result.clip_id[:16]}...", (980, metrics_y + 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        # Legend for trajectory
        cv2.circle(canvas, (main_x + 20, main_y + main_h - 50), 6, (255, 100, 0), -1)
        cv2.putText(canvas, "Predicted", (main_x + 35, main_y + main_h - 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
        cv2.circle(canvas, (main_x + 20, main_y + main_h - 25), 6, (0, 0, 255), -1)
        cv2.putText(canvas, "Ground Truth", (main_x + 35, main_y + main_h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return canvas

    def create_portrait_frame(self, result: ScenarioResult, width=1080, height=1920) -> np.ndarray:
        """9:16 portrait frame for Shorts/Reels"""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (25, 25, 25)

        # Title
        cv2.rectangle(canvas, (0, 0), (width, 120), (35, 35, 35), -1)
        cv2.putText(canvas, "Alpamayo-R1", (width//2 - 150, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(canvas, result.scenario_name_kr, (width//2 - 100, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 255), 2)

        # Main camera view - large
        main_y = 140
        main_h = 600
        main_w = width - 80
        main_x = 40

        if len(result.frames) > 1:
            main_frame = cv2.resize(result.frames[1], (main_w, main_h))
            canvas[main_y:main_y+main_h, main_x:main_x+main_w] = main_frame

        # Trajectory overlay
        self._draw_trajectory(canvas, result, main_x, main_y, main_w, main_h)

        # 4-camera grid below main view
        cam_y = 760
        cam_w, cam_h = 240, 135
        gap = 13
        cam_x_start = (width - 4*cam_w - 3*gap) // 2

        for i in range(min(4, len(result.frames))):
            x = cam_x_start + i * (cam_w + gap)
            frame = cv2.resize(result.frames[i], (cam_w, cam_h))
            canvas[cam_y:cam_y+cam_h, x:x+cam_w] = frame

        # CoC box
        coc_y = 920
        cv2.rectangle(canvas, (40, coc_y), (width-40, coc_y + 140), (40, 40, 40), -1)
        cv2.putText(canvas, "AI Reasoning:", (60, coc_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

        # Word wrap CoC
        coc_words = result.coc.split()
        line1 = " ".join(coc_words[:8])
        line2 = " ".join(coc_words[8:16]) if len(coc_words) > 8 else ""

        cv2.putText(canvas, f'"{line1}', (60, coc_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if line2:
            cv2.putText(canvas, f'{line2}..."', (60, coc_y + 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Metrics - large and centered
        metrics_y = 1100
        cv2.rectangle(canvas, (40, metrics_y), (width-40, metrics_y + 200), (40, 40, 40), -1)

        # ADE
        cv2.putText(canvas, "Prediction Error", (width//2 - 120, metrics_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        cv2.putText(canvas, f"{result.ade:.2f} m", (width//2 - 80, metrics_y + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 255, 100), 3)

        # Inference time
        cv2.putText(canvas, "Inference Time", (width//2 - 100, metrics_y + 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        cv2.putText(canvas, f"{result.inference_time_ms:.0f} ms", (width//2 - 70, metrics_y + 185),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 200, 100), 3)

        # System info bar
        sys_y = 1320
        cv2.rectangle(canvas, (40, sys_y), (width-40, sys_y + 80), (35, 35, 35), -1)
        cv2.putText(canvas, f"GPU: RTX 3090 24GB", (60, sys_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.putText(canvas, f"Model: 10B params (21GB)", (60, sys_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        # Trajectory info
        traj_y = 1420
        cv2.putText(canvas, "6.4 sec trajectory prediction", (width//2 - 180, traj_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
        cv2.putText(canvas, "64 waypoints @ 10Hz", (width//2 - 130, traj_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

        # Footer
        cv2.putText(canvas, "github.com/hwkim3330/carla-alpamayo", (width//2 - 230, height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)

        return canvas

    def _draw_trajectory(self, canvas, result, x_off, y_off, w, h):
        """Draw trajectory on canvas"""
        center_x = x_off + w // 2
        center_y = y_off + h - 30
        scale = 6

        def to_img(xyz):
            return (center_x - int(xyz[1] * scale), center_y - int(xyz[0] * scale))

        # Ground truth (red)
        gt_pts = [to_img(p) for p in result.trajectory_gt[::4]]
        for i in range(len(gt_pts) - 1):
            cv2.line(canvas, gt_pts[i], gt_pts[i+1], (0, 0, 255), 2)

        # Prediction (orange/blue)
        pred_pts = [to_img(p) for p in result.trajectory_pred[::4]]
        for i in range(len(pred_pts) - 1):
            cv2.line(canvas, pred_pts[i], pred_pts[i+1], (255, 100, 0), 3)

        for pt in pred_pts[::2]:
            cv2.circle(canvas, pt, 5, (255, 200, 0), -1)

    def create_intro_landscape(self) -> np.ndarray:
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        cv2.putText(canvas, "NVIDIA Alpamayo-R1", (1920//2 - 350, 400),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)
        cv2.putText(canvas, "Vision-Language-Action Model", (1920//2 - 300, 500),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 2)
        cv2.putText(canvas, "for Autonomous Driving", (1920//2 - 230, 560),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 2)
        cv2.putText(canvas, "Chain-of-Causation Reasoning + Trajectory Prediction",
                   (1920//2 - 420, 680), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 255), 2)
        return canvas

    def create_intro_portrait(self) -> np.ndarray:
        canvas = np.zeros((1920, 1080, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        cv2.putText(canvas, "Alpamayo-R1", (1080//2 - 200, 800),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
        cv2.putText(canvas, "NVIDIA VLA Model", (1080//2 - 170, 900),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 2)
        cv2.putText(canvas, "AI Driving Demo", (1080//2 - 150, 1000),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 200, 255), 2)
        return canvas

    def create_outro_landscape(self) -> np.ndarray:
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        cv2.putText(canvas, "github.com/hwkim3330/carla-alpamayo",
                   (1920//2 - 400, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (100, 200, 255), 2)
        cv2.putText(canvas, "Model: nvidia/Alpamayo-R1-10B",
                   (1920//2 - 250, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)
        cv2.putText(canvas, "Dataset: nvidia/PhysicalAI-Autonomous-Vehicles",
                   (1920//2 - 320, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)
        return canvas

    def create_outro_portrait(self) -> np.ndarray:
        canvas = np.zeros((1920, 1080, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        cv2.putText(canvas, "Follow for more!", (1080//2 - 150, 850),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(canvas, "@hwkim3330", (1080//2 - 100, 950),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 255), 2)
        return canvas

    def generate_video(self, scenarios, output_path, mode="landscape", fps=30,
                       intro_sec=3, scenario_sec=6, outro_sec=3):

        if mode == "landscape":
            w, h = 1920, 1080
            create_frame = self.create_landscape_frame
            intro = self.create_intro_landscape()
            outro = self.create_outro_landscape()
        else:
            w, h = 1080, 1920
            create_frame = lambda r: self.create_portrait_frame(r)
            intro = self.create_intro_portrait()
            outro = self.create_outro_portrait()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # Intro
        print(f"Generating {mode} video...")
        for _ in range(int(intro_sec * fps)):
            out.write(cv2.cvtColor(intro, cv2.COLOR_RGB2BGR))

        # Scenarios
        for clip_id, name, name_kr in scenarios:
            result = self.run_inference(clip_id, name, name_kr)
            if result is None:
                continue

            frame = create_frame(result)
            for _ in range(int(scenario_sec * fps)):
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Outro
        for _ in range(int(outro_sec * fps)):
            out.write(cv2.cvtColor(outro, cv2.COLOR_RGB2BGR))

        out.release()

        # Convert to H.264
        h264_path = output_path.replace('.mp4', '_h264.mp4')
        os.system(f'ffmpeg -y -i {output_path} -c:v libx264 -preset medium -crf 23 {h264_path} 2>/dev/null')
        if os.path.exists(h264_path):
            os.replace(h264_path, output_path)

        print(f"Saved: {output_path}")


def main():
    import pandas as pd

    clip_ids = pd.read_parquet(
        "/mnt/data/lfm_agi/alpamayo_code/notebooks/clip_ids.parquet"
    )["clip_id"].tolist()

    # Scenarios: (clip_id, english_name, korean_name)
    scenarios = [
        (clip_ids[774], "Construction Zone", "공사 구간"),
        (clip_ids[100], "Following Vehicle", "앞차 추종"),
        (clip_ids[400], "Green Light", "녹색 신호"),
        (clip_ids[0], "Parked Vehicle", "주차 차량 회피"),
        (clip_ids[200], "Lead Vehicle", "선행 차량"),
        (clip_ids[500], "Urban Driving", "도심 주행"),
        (clip_ids[300], "Highway", "고속도로"),
        (clip_ids[600], "Intersection", "교차로"),
    ]

    generator = VideoGeneratorV2()

    # Generate landscape video (YouTube)
    generator.generate_video(
        scenarios=scenarios,
        output_path=str(OUTPUT_DIR / "alpamayo_demo_v2.mp4"),
        mode="landscape",
        intro_sec=3,
        scenario_sec=5,
        outro_sec=3
    )

    # Generate portrait video (Shorts)
    generator.generate_video(
        scenarios=scenarios[:5],  # Shorter for shorts
        output_path=str(OUTPUT_DIR / "alpamayo_shorts.mp4"),
        mode="portrait",
        intro_sec=2,
        scenario_sec=4,
        outro_sec=2
    )

    print("\nAll videos generated!")


if __name__ == "__main__":
    main()
