#!/usr/bin/env python3
"""
Continuous Scenario Video with TTS

Creates a video showing continuous inference over time,
simulating real-time autonomous driving with AI narration.
"""

import os
import sys
import copy
import time
import asyncio
import numpy as np
import cv2
from pathlib import Path
import torch
import edge_tts
import subprocess

sys.path.insert(0, "/mnt/data/lfm_agi/alpamayo_code/src")

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

OUTPUT_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/output")
TEMP_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/temp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


class ContinuousVideoGenerator:
    def __init__(self):
        self.model = None
        self.processor = None
        self.tts_voice = "en-US-ChristopherNeural"  # Professional male voice

    def load_model(self):
        if self.model is None:
            print("Loading Alpamayo model...")
            self.model = AlpamayoR1.from_pretrained(
                "nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16
            ).to("cuda")
            self.processor = helper.get_processor(self.model.tokenizer)
            print("Model loaded.")

    async def generate_tts(self, text: str, output_path: str):
        """Generate TTS audio file"""
        communicate = edge_tts.Communicate(text, self.tts_voice)
        await communicate.save(output_path)

    def run_inference_at_time(self, clip_id: str, t0_us: int):
        """Run inference at specific timestamp"""
        self.load_model()

        try:
            data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
            frames = data["image_frames"].permute(0, 1, 3, 4, 2).numpy()  # [T, C, H, W, 3]

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

            torch.cuda.synchronize()
            start = time.time()

            torch.cuda.manual_seed_all(42)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                    data=copy.deepcopy(model_inputs),
                    top_p=0.98, temperature=0.6,
                    num_traj_samples=1, max_generation_length=256,
                    return_extra=True,
                )

            torch.cuda.synchronize()
            inference_ms = (time.time() - start) * 1000

            coc = extra["cot"][0][0][0]
            gt = data["ego_future_xyz"].cpu()[0, 0].numpy()
            pred = pred_xyz.cpu()[0, 0, 0].numpy()
            ade = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1).mean()

            return {
                "frames": frames,
                "coc": coc,
                "ade": ade,
                "inference_ms": inference_ms,
                "pred": pred,
                "gt": gt,
                "t0_us": t0_us
            }
        except Exception as e:
            print(f"Error at t0={t0_us}: {e}")
            return None

    def create_frame(self, result, width=1920, height=1080, show_processing=False):
        """Create video frame with 4-camera layout"""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        frames = result["frames"]  # [T, C, H, W, 3]

        # Use latest timestamp, 4 cameras
        t_idx = frames.shape[0] - 1

        # 2x2 camera grid - upper portion
        cam_positions = [
            (0, "Front Left", 40, 100),
            (1, "Front Center", 500, 100),
            (2, "Front Right", 960, 100),
            (3, "Rear", 1420, 100),
        ]

        cam_w, cam_h = 440, 248

        for cam_idx, label, x, y in cam_positions:
            if cam_idx < frames.shape[1]:
                frame = frames[t_idx, cam_idx]
                frame_resized = cv2.resize(frame.astype(np.uint8), (cam_w, cam_h))
                canvas[y:y+cam_h, x:x+cam_w] = frame_resized

                # Label
                cv2.rectangle(canvas, (x, y), (x + 120, y + 25), (0, 0, 0), -1)
                cv2.putText(canvas, label, (x + 5, y + 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Main view with trajectory - center bottom
        main_x, main_y = 40, 380
        main_w, main_h = 900, 500

        if frames.shape[1] > 1:
            main_frame = cv2.resize(frames[t_idx, 1].astype(np.uint8), (main_w, main_h))
            canvas[main_y:main_y+main_h, main_x:main_x+main_w] = main_frame

        # Draw trajectory
        self._draw_trajectory(canvas, result["pred"], result["gt"],
                             main_x, main_y, main_w, main_h)

        # Title bar
        cv2.rectangle(canvas, (0, 0), (width, 80), (30, 30, 30), -1)
        cv2.putText(canvas, "Alpamayo-R1: Real-time Autonomous Driving AI",
                   (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

        # Status indicator
        if show_processing:
            cv2.circle(canvas, (width - 100, 45), 15, (0, 165, 255), -1)
            cv2.putText(canvas, "THINKING", (width - 200, 52),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            cv2.circle(canvas, (width - 100, 45), 15, (0, 255, 0), -1)
            cv2.putText(canvas, "READY", (width - 180, 52),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Info panel - right side
        panel_x, panel_y = 980, 380
        panel_w, panel_h = 900, 500
        cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                     (35, 35, 35), -1)

        # CoC Section
        cv2.putText(canvas, "AI Reasoning (Chain-of-Causation):",
                   (panel_x + 20, panel_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

        # Word wrap CoC
        coc = result["coc"]
        words = coc.split()
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            if len(" ".join(current_line)) > 50:
                lines.append(" ".join(current_line[:-1]))
                current_line = [word]
        if current_line:
            lines.append(" ".join(current_line))

        for i, line in enumerate(lines[:4]):
            cv2.putText(canvas, line, (panel_x + 20, panel_y + 80 + i * 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Metrics
        metrics_y = panel_y + 250
        cv2.line(canvas, (panel_x + 20, metrics_y - 20), (panel_x + panel_w - 20, metrics_y - 20),
                (60, 60, 60), 1)

        cv2.putText(canvas, "Performance Metrics", (panel_x + 20, metrics_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

        cv2.putText(canvas, f"Prediction Error (ADE): {result['ade']:.2f} meters",
                   (panel_x + 20, metrics_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

        cv2.putText(canvas, f"Inference Time: {result['inference_ms']:.0f} ms",
                   (panel_x + 20, metrics_y + 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)

        cv2.putText(canvas, f"Trajectory: 6.4 seconds (64 waypoints @ 10Hz)",
                   (panel_x + 20, metrics_y + 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # System info
        sys_y = metrics_y + 170
        cv2.line(canvas, (panel_x + 20, sys_y - 20), (panel_x + panel_w - 20, sys_y - 20),
                (60, 60, 60), 1)

        cv2.putText(canvas, "System: NVIDIA RTX 3090 (24GB) | Model: Alpamayo-R1-10B (21GB)",
                   (panel_x + 20, sys_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        # Legend
        cv2.circle(canvas, (main_x + 20, main_y + main_h - 40), 8, (255, 100, 0), -1)
        cv2.putText(canvas, "AI Predicted Path", (main_x + 35, main_y + main_h - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

        cv2.circle(canvas, (main_x + 200, main_y + main_h - 40), 8, (0, 0, 255), -1)
        cv2.putText(canvas, "Actual Path", (main_x + 215, main_y + main_h - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return canvas

    def _draw_trajectory(self, canvas, pred, gt, x_off, y_off, w, h):
        """Draw trajectory overlay"""
        cx, cy = x_off + w // 2, y_off + h - 30
        scale = 5

        def to_img(xyz):
            return (cx - int(xyz[1] * scale), cy - int(xyz[0] * scale))

        # GT (red)
        pts = [to_img(p) for p in gt[::4]]
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i+1], (0, 0, 255), 2)

        # Pred (orange)
        pts = [to_img(p) for p in pred[::4]]
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i+1], (255, 100, 0), 3)
        for pt in pts[::2]:
            cv2.circle(canvas, pt, 5, (255, 200, 0), -1)

    def create_intro(self, width=1920, height=1080):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (15, 15, 15)

        cv2.putText(canvas, "NVIDIA Alpamayo-R1", (width//2 - 320, height//2 - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 3)
        cv2.putText(canvas, "Continuous Autonomous Driving Demo",
                   (width//2 - 350, height//2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 200, 255), 2)
        cv2.putText(canvas, "Real-time AI Reasoning + Trajectory Prediction",
                   (width//2 - 340, height//2 + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)
        return canvas

    def create_outro(self, width=1920, height=1080):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (15, 15, 15)

        cv2.putText(canvas, "github.com/hwkim3330/carla-alpamayo",
                   (width//2 - 380, height//2 - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (100, 200, 255), 2)
        cv2.putText(canvas, "Model: nvidia/Alpamayo-R1-10B",
                   (width//2 - 240, height//2 + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)
        cv2.putText(canvas, "Subscribe for more AI demos!",
                   (width//2 - 220, height//2 + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return canvas

    async def generate_video(self, clip_id: str, timestamps_us: list,
                             output_path: str, fps: int = 30):
        """Generate continuous video with TTS"""

        width, height = 1920, 1080
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # First, run all inferences and generate TTS
        print("Running inferences and generating TTS...")
        results = []
        tts_files = []

        for i, t0 in enumerate(timestamps_us):
            print(f"Inference {i+1}/{len(timestamps_us)} at t0={t0}")
            result = self.run_inference_at_time(clip_id, t0)
            if result:
                results.append(result)

                # Generate TTS for CoC
                tts_text = f"AI reasoning: {result['coc']}"
                tts_path = str(TEMP_DIR / f"tts_{i:03d}.mp3")
                await self.generate_tts(tts_text, tts_path)
                tts_files.append(tts_path)

                print(f"  CoC: {result['coc'][:50]}...")

        if not results:
            print("No results to generate video")
            return

        # Create video without audio first
        video_no_audio = str(TEMP_DIR / "video_no_audio.mp4")
        out = cv2.VideoWriter(video_no_audio, fourcc, fps, (width, height))

        # Intro (3 sec)
        print("Generating intro...")
        intro = self.create_intro()
        for _ in range(fps * 3):
            out.write(cv2.cvtColor(intro, cv2.COLOR_RGB2BGR))

        # Each result segment
        segment_duration = 4  # seconds per inference result

        for i, result in enumerate(results):
            print(f"Generating segment {i+1}/{len(results)}...")

            # Show "processing" for first 0.5 sec
            for _ in range(int(fps * 0.5)):
                frame = self.create_frame(result, show_processing=True)
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Show result for remaining time
            for _ in range(int(fps * (segment_duration - 0.5))):
                frame = self.create_frame(result, show_processing=False)
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Outro (3 sec)
        print("Generating outro...")
        outro = self.create_outro()
        for _ in range(fps * 3):
            out.write(cv2.cvtColor(outro, cv2.COLOR_RGB2BGR))

        out.release()

        # Concatenate TTS audio files
        print("Combining audio...")
        audio_list_file = str(TEMP_DIR / "audio_list.txt")
        combined_audio = str(TEMP_DIR / "combined_audio.mp3")

        # Create silence for intro (3 sec)
        intro_silence = str(TEMP_DIR / "intro_silence.mp3")
        os.system(f'ffmpeg -y -f lavfi -i anullsrc=r=44100:cl=mono -t 3 {intro_silence} 2>/dev/null')

        with open(audio_list_file, 'w') as f:
            f.write(f"file '{intro_silence}'\n")
            for tts_file in tts_files:
                f.write(f"file '{tts_file}'\n")
                # Add padding between segments
                silence_pad = str(TEMP_DIR / "silence_pad.mp3")
                os.system(f'ffmpeg -y -f lavfi -i anullsrc=r=44100:cl=mono -t 1 {silence_pad} 2>/dev/null')
                f.write(f"file '{silence_pad}'\n")

        os.system(f'ffmpeg -y -f concat -safe 0 -i {audio_list_file} -c copy {combined_audio} 2>/dev/null')

        # Combine video and audio
        print("Combining video and audio...")
        final_output = output_path.replace('.mp4', '_final.mp4')
        os.system(f'ffmpeg -y -i {video_no_audio} -i {combined_audio} -c:v libx264 -c:a aac -shortest {final_output} 2>/dev/null')

        if os.path.exists(final_output):
            os.replace(final_output, output_path)
            print(f"Saved: {output_path}")
        else:
            # Fallback: just convert video without audio
            os.system(f'ffmpeg -y -i {video_no_audio} -c:v libx264 {output_path} 2>/dev/null')
            print(f"Saved (no audio): {output_path}")

        # Cleanup
        for f in TEMP_DIR.glob("*"):
            try:
                f.unlink()
            except:
                pass


async def main():
    import pandas as pd

    clip_ids = pd.read_parquet(
        "/mnt/data/lfm_agi/alpamayo_code/notebooks/clip_ids.parquet"
    )["clip_id"].tolist()

    # Use a clip with multiple timestamps available
    clip_id = clip_ids[774]  # Construction zone clip

    # Generate timestamps (2 second intervals in microseconds)
    # Start from 2M us (2 sec) since history needs some data
    timestamps = [2_000_000 + i * 2_000_000 for i in range(6)]  # 6 inferences

    generator = ContinuousVideoGenerator()

    output_path = str(OUTPUT_DIR / "alpamayo_continuous.mp4")
    await generator.generate_video(clip_id, timestamps, output_path)

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
