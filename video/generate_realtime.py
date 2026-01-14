#!/usr/bin/env python3
"""
Realistic Real-time Driving Video with TTS

- Uses T dimension for frame animation (moving video)
- Continuous inference across timestamps
- Professional TTS narration
- Proper camera layout
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

sys.path.insert(0, "/mnt/data/lfm_agi/alpamayo_code/src")

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

OUTPUT_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/output")
TEMP_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/temp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


class RealtimeVideoGenerator:
    def __init__(self):
        self.model = None
        self.processor = None
        self.tts_voice = "en-US-GuyNeural"  # Clear male voice

    def load_model(self):
        if self.model is None:
            print("Loading Alpamayo model...")
            self.model = AlpamayoR1.from_pretrained(
                "nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16
            ).to("cuda")
            self.processor = helper.get_processor(self.model.tokenizer)

    def run_inference(self, clip_id: str, t0_us: int):
        """Run inference and get frames + results"""
        self.load_model()

        try:
            data = load_physical_aiavdataset(clip_id, t0_us=t0_us)

            # frames: [T, C, 3, H, W] -> we use T for animation
            # T = 4 time steps, C = 4 cameras
            frames = data["image_frames"]  # [4, 4, 3, 1080, 1920]

            messages = helper.create_message(frames.flatten(0, 1))
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

            # Convert frames: [T, C, 3, H, W] -> list of numpy arrays
            # Use camera index 0 (main front camera) across time
            animated_frames = []
            for t in range(frames.shape[0]):
                # Use first camera (index 0) for main view
                frame = frames[t, 0].permute(1, 2, 0).numpy().astype(np.uint8)
                animated_frames.append(frame)

            return {
                "frames": animated_frames,  # 4 frames for animation
                "all_frames": frames,  # Full tensor for multi-cam view
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

    def create_animated_frame(self, result, frame_idx, width=1920, height=1080):
        """Create a single frame with animation"""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 18)

        all_frames = result["all_frames"]  # [T, C, 3, H, W]
        T, C = all_frames.shape[:2]

        # Map frame_idx to time step (loop through T dimension)
        t_idx = frame_idx % T

        # === MAIN VIEW (large, center-left) ===
        main_frame = all_frames[t_idx, 0].permute(1, 2, 0).numpy().astype(np.uint8)
        main_h, main_w = 600, 1000
        main_x, main_y = 40, 100

        main_resized = cv2.resize(main_frame, (main_w, main_h))
        canvas[main_y:main_y+main_h, main_x:main_x+main_w] = main_resized

        # Draw trajectory on main view
        self._draw_trajectory(canvas, result["pred"], result["gt"],
                             main_x, main_y, main_w, main_h)

        # === SMALL CAMERA VIEWS (right column) ===
        # Show different time steps as "multi-camera" feel
        small_w, small_h = 400, 225
        small_x = 1080

        cam_labels = [f"t-{3-t_idx}00ms" if t_idx > 0 else "Current",
                     f"t-{3-(t_idx+1)%T}00ms",
                     f"t-{3-(t_idx+2)%T}00ms",
                     f"t-{3-(t_idx+3)%T}00ms"]

        for i in range(4):
            y = 100 + i * (small_h + 10)
            ti = (t_idx + i) % T

            small_frame = all_frames[ti, 0].permute(1, 2, 0).numpy().astype(np.uint8)
            small_resized = cv2.resize(small_frame, (small_w, small_h))
            canvas[y:y+small_h, small_x:small_x+small_w] = small_resized

            # Highlight current frame
            if i == 0:
                cv2.rectangle(canvas, (small_x-2, y-2),
                             (small_x+small_w+2, y+small_h+2), (0, 255, 0), 2)
                cv2.putText(canvas, "LIVE", (small_x + small_w - 60, y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # === TITLE BAR ===
        cv2.rectangle(canvas, (0, 0), (width, 80), (25, 25, 25), -1)
        cv2.putText(canvas, "NVIDIA Alpamayo-R1 | Autonomous Driving AI",
                   (40, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Status indicator
        cv2.circle(canvas, (width - 80, 45), 12, (0, 255, 0), -1)
        cv2.putText(canvas, "LIVE", (width - 150, 52),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # === COC BOX (bottom) ===
        coc_y = 720
        cv2.rectangle(canvas, (40, coc_y), (1040, coc_y + 120), (30, 30, 30), -1)
        cv2.rectangle(canvas, (40, coc_y), (1040, coc_y + 120), (60, 60, 60), 2)

        cv2.putText(canvas, "AI Reasoning:", (60, coc_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

        # Word wrap CoC
        words = result["coc"].split()
        line1 = " ".join(words[:10])
        line2 = " ".join(words[10:20]) if len(words) > 10 else ""

        cv2.putText(canvas, f'"{line1}', (60, coc_y + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        if line2:
            cv2.putText(canvas, f'{line2}..."', (60, coc_y + 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # === METRICS (bottom right) ===
        metrics_y = 720
        cv2.rectangle(canvas, (1080, metrics_y), (width - 40, metrics_y + 120),
                     (30, 30, 30), -1)

        cv2.putText(canvas, f"ADE: {result['ade']:.2f}m", (1100, metrics_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        cv2.putText(canvas, f"Inference: {result['inference_ms']:.0f}ms",
                   (1100, metrics_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   (255, 200, 100), 2)
        cv2.putText(canvas, "RTX 3090 | 10B params", (1100, metrics_y + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        # === TRAJECTORY LEGEND ===
        cv2.circle(canvas, (main_x + 20, main_y + main_h - 50), 8, (255, 100, 0), -1)
        cv2.putText(canvas, "AI Prediction", (main_x + 35, main_y + main_h - 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
        cv2.circle(canvas, (main_x + 180, main_y + main_h - 50), 8, (0, 0, 255), -1)
        cv2.putText(canvas, "Ground Truth", (main_x + 195, main_y + main_h - 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Progress bar showing inference cycle
        progress = (frame_idx % 60) / 60.0  # 2 sec cycle at 30fps
        bar_y = 860
        cv2.rectangle(canvas, (40, bar_y), (width - 40, bar_y + 8), (40, 40, 40), -1)
        cv2.rectangle(canvas, (40, bar_y), (int(40 + (width - 80) * progress), bar_y + 8),
                     (100, 200, 255), -1)
        cv2.putText(canvas, "Inference Cycle (2 sec)", (40, bar_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

        return canvas

    def _draw_trajectory(self, canvas, pred, gt, x_off, y_off, w, h):
        """Draw trajectory overlay"""
        cx, cy = x_off + w // 2, y_off + h - 50
        scale = 6

        def to_img(xyz):
            return (cx - int(xyz[1] * scale), cy - int(xyz[0] * scale))

        # GT (red, thinner)
        pts = [to_img(p) for p in gt[::4]]
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i+1], (0, 0, 255), 2)

        # Pred (orange, thicker)
        pts = [to_img(p) for p in pred[::4]]
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i+1], (255, 100, 0), 3)
        for pt in pts[::2]:
            cv2.circle(canvas, pt, 6, (255, 200, 0), -1)

    async def generate_tts(self, text: str, output_path: str):
        """Generate TTS audio"""
        communicate = edge_tts.Communicate(text, self.tts_voice)
        await communicate.save(output_path)

    def create_intro(self, width=1920, height=1080):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (12, 12, 12)

        cv2.putText(canvas, "NVIDIA Alpamayo-R1", (width//2 - 320, height//2 - 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)
        cv2.putText(canvas, "Real-time Autonomous Driving AI",
                   (width//2 - 320, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                   (100, 200, 255), 2)
        cv2.putText(canvas, "Chain-of-Causation Reasoning | Trajectory Prediction",
                   (width//2 - 380, height//2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                   (150, 150, 150), 2)
        return canvas

    def create_outro(self, width=1920, height=1080):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (12, 12, 12)

        cv2.putText(canvas, "github.com/hwkim3330/carla-alpamayo",
                   (width//2 - 380, height//2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                   (100, 200, 255), 2)
        cv2.putText(canvas, "Subscribe for more AI demos!",
                   (width//2 - 230, height//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                   (255, 255, 255), 2)
        return canvas

    async def generate_video(self, clip_id: str, timestamps_us: list,
                             output_path: str, fps: int = 30):
        """Generate continuous realistic video with TTS"""

        width, height = 1920, 1080
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Run all inferences first
        print("Running inferences...")
        results = []
        tts_texts = []

        for i, t0 in enumerate(timestamps_us):
            print(f"Inference {i+1}/{len(timestamps_us)} at t0={t0}")
            result = self.run_inference(clip_id, t0)
            if result:
                results.append(result)
                tts_texts.append(result["coc"])
                print(f"  CoC: {result['coc'][:50]}...")
                print(f"  ADE: {result['ade']:.2f}m, Time: {result['inference_ms']:.0f}ms")

        if not results:
            print("No results!")
            return

        # Generate TTS files
        print("\nGenerating TTS audio...")
        tts_files = []
        for i, text in enumerate(tts_texts):
            tts_path = str(TEMP_DIR / f"tts_{i:03d}.mp3")
            await self.generate_tts(f"{text}", tts_path)
            tts_files.append(tts_path)

        # Create video
        print("\nGenerating video frames...")
        video_no_audio = str(TEMP_DIR / "video_no_audio.mp4")
        out = cv2.VideoWriter(video_no_audio, fourcc, fps, (width, height))

        # Intro (3 sec)
        intro = self.create_intro()
        for _ in range(fps * 3):
            out.write(cv2.cvtColor(intro, cv2.COLOR_RGB2BGR))

        # Each inference result (animated, ~4 sec each)
        frames_per_result = fps * 4  # 4 seconds per inference

        for result_idx, result in enumerate(results):
            print(f"Generating segment {result_idx + 1}/{len(results)}...")

            for frame_idx in range(frames_per_result):
                frame = self.create_animated_frame(result, frame_idx)
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Outro (3 sec)
        outro = self.create_outro()
        for _ in range(fps * 3):
            out.write(cv2.cvtColor(outro, cv2.COLOR_RGB2BGR))

        out.release()

        # Combine audio
        print("\nCombining audio...")

        # Create silence files
        intro_silence = str(TEMP_DIR / "intro_silence.mp3")
        segment_silence = str(TEMP_DIR / "segment_silence.mp3")
        os.system(f'ffmpeg -y -f lavfi -i anullsrc=r=44100:cl=mono -t 3 -c:a libmp3lame {intro_silence} 2>/dev/null')
        os.system(f'ffmpeg -y -f lavfi -i anullsrc=r=44100:cl=mono -t 1 -c:a libmp3lame {segment_silence} 2>/dev/null')

        # Concat audio
        audio_list = str(TEMP_DIR / "audio_list.txt")
        with open(audio_list, 'w') as f:
            f.write(f"file '{intro_silence}'\n")
            for tts_file in tts_files:
                f.write(f"file '{tts_file}'\n")
                f.write(f"file '{segment_silence}'\n")

        combined_audio = str(TEMP_DIR / "combined.mp3")
        os.system(f'ffmpeg -y -f concat -safe 0 -i {audio_list} -c:a libmp3lame {combined_audio} 2>/dev/null')

        # Final merge
        print("Merging video and audio...")
        os.system(f'ffmpeg -y -i {video_no_audio} -i {combined_audio} -c:v libx264 -c:a aac -shortest {output_path} 2>/dev/null')

        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            # Fallback without audio
            os.system(f'ffmpeg -y -i {video_no_audio} -c:v libx264 {output_path} 2>/dev/null')

        print(f"\nSaved: {output_path}")

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

    # Use construction zone clip
    clip_id = clip_ids[774]

    # Timestamps: every 2 seconds (2M microseconds)
    timestamps = [2_000_000 + i * 2_000_000 for i in range(8)]  # 8 inferences

    generator = RealtimeVideoGenerator()
    output_path = str(OUTPUT_DIR / "alpamayo_realtime.mp4")

    await generator.generate_video(clip_id, timestamps, output_path)
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
