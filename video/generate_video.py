#!/usr/bin/env python3
"""
Alpamayo Demo Video Generator

다양한 시나리오의 추론 결과를 편집하여 데모 영상 생성
"""

import os
import sys
import copy
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch

# Add alpamayo to path
sys.path.insert(0, "/mnt/data/lfm_agi/alpamayo_code/src")

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

# Paths
OUTPUT_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/output")
CLIPS_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/clips")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ScenarioResult:
    """Single scenario inference result"""
    clip_id: str
    scenario_name: str
    coc: str
    ade: float
    frames: np.ndarray  # [T, H, W, C]
    trajectory_pred: np.ndarray  # [64, 3]
    trajectory_gt: np.ndarray  # [64, 3]


class VideoGenerator:
    """Generate demo video from multiple scenarios"""

    def __init__(self,
                 width: int = 1920,
                 height: int = 1080,
                 fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.model = None
        self.processor = None

        # Try to load a nice font, fallback to default
        self.font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    def load_model(self):
        """Load Alpamayo model"""
        if self.model is None:
            print("Loading Alpamayo model...")
            self.model = AlpamayoR1.from_pretrained(
                "nvidia/Alpamayo-R1-10B",
                dtype=torch.bfloat16
            ).to("cuda")
            self.processor = helper.get_processor(self.model.tokenizer)
            print("Model loaded.")

    def run_inference(self, clip_id: str, scenario_name: str) -> Optional[ScenarioResult]:
        """Run inference on a single clip"""
        self.load_model()

        try:
            print(f"Processing: {scenario_name} ({clip_id[:8]}...)")
            data = load_physical_aiavdataset(clip_id)

            # Get frames
            frames = data["image_frames"].flatten(0, 1).permute(0, 2, 3, 1).numpy()

            # Run inference
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

            torch.cuda.manual_seed_all(42)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                    data=copy.deepcopy(model_inputs),
                    top_p=0.98, temperature=0.6,
                    num_traj_samples=1, max_generation_length=256,
                    return_extra=True,
                )

            coc = extra["cot"][0][0][0]
            gt_xy = data["ego_future_xyz"].cpu()[0, 0].numpy()
            pred_xy = pred_xyz.cpu()[0, 0, 0].numpy()

            ade = np.linalg.norm(pred_xy[:, :2] - gt_xy[:, :2], axis=1).mean()

            return ScenarioResult(
                clip_id=clip_id,
                scenario_name=scenario_name,
                coc=coc,
                ade=ade,
                frames=frames,
                trajectory_pred=pred_xy,
                trajectory_gt=gt_xy
            )

        except Exception as e:
            print(f"Error processing {clip_id}: {e}")
            return None

    def create_scenario_frame(self,
                              result: ScenarioResult,
                              frame_idx: int = 0) -> np.ndarray:
        """Create a single frame for video with all overlays"""

        # Create base canvas
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)  # Dark gray background

        # Main view (front camera) - top center
        main_frame = result.frames[frame_idx % len(result.frames)]
        main_h, main_w = 540, 960
        main_frame_resized = cv2.resize(main_frame, (main_w, main_h))

        x_offset = (self.width - main_w) // 2
        y_offset = 50
        canvas[y_offset:y_offset+main_h, x_offset:x_offset+main_w] = main_frame_resized

        # Draw trajectory on main view
        canvas = self._draw_trajectory_overlay(
            canvas, result,
            x_offset, y_offset, main_w, main_h
        )

        # Thumbnail cameras - bottom row
        thumb_size = (240, 135)
        thumb_y = 620
        thumb_spacing = 20
        total_thumb_width = 4 * thumb_size[0] + 3 * thumb_spacing
        thumb_x_start = (self.width - total_thumb_width) // 2

        for i in range(min(4, len(result.frames))):
            thumb = cv2.resize(result.frames[i], thumb_size)
            x = thumb_x_start + i * (thumb_size[0] + thumb_spacing)
            canvas[thumb_y:thumb_y+thumb_size[1], x:x+thumb_size[0]] = thumb

            # Camera label
            cv2.putText(canvas, f"CAM {i+1}", (x + 5, thumb_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # CoC text box - bottom
        coc_box_y = 780
        coc_box_h = 80
        cv2.rectangle(canvas, (100, coc_box_y), (self.width-100, coc_box_y+coc_box_h),
                     (50, 50, 50), -1)
        cv2.rectangle(canvas, (100, coc_box_y), (self.width-100, coc_box_y+coc_box_h),
                     (100, 100, 100), 2)

        # CoC icon and text
        cv2.putText(canvas, "Chain-of-Causation:", (120, coc_box_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        # Wrap CoC text if too long
        coc_display = result.coc[:100] + "..." if len(result.coc) > 100 else result.coc
        cv2.putText(canvas, f'"{coc_display}"', (120, coc_box_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Metrics bar - very bottom
        metrics_y = 880
        cv2.putText(canvas, f"ADE: {result.ade:.2f}m", (100, metrics_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        cv2.putText(canvas, f"|  Scenario: {result.scenario_name}", (280, metrics_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        cv2.putText(canvas, f"|  Clip: {result.clip_id[:12]}...", (600, metrics_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Title
        cv2.putText(canvas, "NVIDIA Alpamayo-R1 Demo", (self.width//2 - 200, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        return canvas

    def _draw_trajectory_overlay(self, canvas, result, x_off, y_off, w, h):
        """Draw trajectory prediction overlay on main view"""

        # Convert trajectory to image coordinates
        # Assuming ego-centric: x=forward, y=left
        # Map to image: center-bottom of main view

        center_x = x_off + w // 2
        center_y = y_off + h - 50  # Bottom of main view

        scale = 8  # pixels per meter

        def traj_to_img(xyz):
            # x (forward) -> up in image (negative y)
            # y (left) -> left in image (negative x)
            img_x = center_x - int(xyz[1] * scale)
            img_y = center_y - int(xyz[0] * scale)
            return (img_x, img_y)

        # Draw GT trajectory (red)
        gt_points = [traj_to_img(p) for p in result.trajectory_gt[::4]]  # Every 4th point
        for i in range(len(gt_points) - 1):
            cv2.line(canvas, gt_points[i], gt_points[i+1], (0, 0, 255), 2)

        # Draw predicted trajectory (blue)
        pred_points = [traj_to_img(p) for p in result.trajectory_pred[::4]]
        for i in range(len(pred_points) - 1):
            cv2.line(canvas, pred_points[i], pred_points[i+1], (255, 100, 0), 3)

        # Draw waypoints
        for pt in pred_points[::2]:
            cv2.circle(canvas, pt, 4, (255, 200, 0), -1)

        # Legend
        cv2.putText(canvas, "Predicted", (x_off + 10, y_off + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
        cv2.putText(canvas, "Ground Truth", (x_off + 10, y_off + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return canvas

    def create_intro_frame(self) -> np.ndarray:
        """Create intro title frame"""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        # Title
        cv2.putText(canvas, "NVIDIA Alpamayo-R1",
                   (self.width//2 - 300, self.height//2 - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)

        cv2.putText(canvas, "Vision-Language-Action Model for Autonomous Driving",
                   (self.width//2 - 400, self.height//2 + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

        cv2.putText(canvas, "Chain-of-Causation Reasoning + Trajectory Prediction",
                   (self.width//2 - 350, self.height//2 + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

        return canvas

    def create_outro_frame(self) -> np.ndarray:
        """Create outro frame"""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        cv2.putText(canvas, "github.com/hwkim3330/carla-alpamayo",
                   (self.width//2 - 350, self.height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 200, 255), 2)

        cv2.putText(canvas, "Model: nvidia/Alpamayo-R1-10B",
                   (self.width//2 - 220, self.height//2 + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

        cv2.putText(canvas, "Dataset: nvidia/PhysicalAI-Autonomous-Vehicles",
                   (self.width//2 - 280, self.height//2 + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

        return canvas

    def generate_video(self,
                       scenarios: List[Tuple[str, str]],
                       output_path: str,
                       intro_duration: float = 3.0,
                       scenario_duration: float = 5.0,
                       outro_duration: float = 3.0):
        """
        Generate full demo video

        Args:
            scenarios: List of (clip_id, scenario_name) tuples
            output_path: Output video file path
            intro_duration: Intro length in seconds
            scenario_duration: Each scenario length in seconds
            outro_duration: Outro length in seconds
        """

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        # Intro
        print("Generating intro...")
        intro_frame = self.create_intro_frame()
        for _ in range(int(intro_duration * self.fps)):
            out.write(cv2.cvtColor(intro_frame, cv2.COLOR_RGB2BGR))

        # Process each scenario
        for clip_id, scenario_name in scenarios:
            result = self.run_inference(clip_id, scenario_name)
            if result is None:
                continue

            print(f"Generating frames for: {scenario_name}")

            # Generate frames for this scenario
            num_frames = int(scenario_duration * self.fps)
            for i in range(num_frames):
                # Cycle through camera frames
                frame_idx = (i // 10) % len(result.frames)
                frame = self.create_scenario_frame(result, frame_idx)
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Outro
        print("Generating outro...")
        outro_frame = self.create_outro_frame()
        for _ in range(int(outro_duration * self.fps)):
            out.write(cv2.cvtColor(outro_frame, cv2.COLOR_RGB2BGR))

        out.release()
        print(f"Video saved to: {output_path}")

        # Convert to H.264 for better compatibility
        h264_path = output_path.replace('.mp4', '_h264.mp4')
        os.system(f'ffmpeg -y -i {output_path} -c:v libx264 -preset medium -crf 23 {h264_path} 2>/dev/null')
        if os.path.exists(h264_path):
            os.replace(h264_path, output_path)
            print(f"Converted to H.264: {output_path}")


def main():
    """Main entry point"""
    import pandas as pd

    # Load clip IDs
    clip_ids = pd.read_parquet(
        "/mnt/data/lfm_agi/alpamayo_code/notebooks/clip_ids.parquet"
    )["clip_id"].tolist()

    # Define scenarios to include
    # Format: (clip_id, scenario_name)
    scenarios = [
        (clip_ids[774], "Construction Zone"),      # 공사 구간
        (clip_ids[100], "Following Vehicle"),      # 앞차 추종
        (clip_ids[400], "Green Light"),            # 녹색 신호
        (clip_ids[0], "Parked Vehicle"),           # 주차 차량
        (clip_ids[200], "Lead Vehicle"),           # 선행 차량
        (clip_ids[500], "Urban Driving"),          # 도심 주행
    ]

    # Generate video
    generator = VideoGenerator()
    output_path = str(OUTPUT_DIR / "alpamayo_demo.mp4")

    generator.generate_video(
        scenarios=scenarios,
        output_path=output_path,
        intro_duration=3.0,
        scenario_duration=5.0,
        outro_duration=3.0
    )

    print("\nDone!")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
