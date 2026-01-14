#!/usr/bin/env python3
"""
Final Alpamayo demo video with:
- Correct camera labels (Cross Left, Front Wide, Cross Right, Front Tele)
- 3-second inference intervals with trajectory visualization
- hwkim3330 watermark (semi-transparent)
- TTS narration using gTTS
"""

import pickle
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap
import subprocess
import tempfile
from gtts import gTTS
import os

# Paths
CACHE_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/cache")
OUTPUT_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Video settings
WIDTH = 1920
HEIGHT = 1080
FPS = 10
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_scenario(pkl_path):
    """Load cached scenario data."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def create_quad_view(cam_frames):
    """Create 2x2 grid view from 4 camera frames."""
    # cam_frames: (4, H, W, 3) - Cross Left, Front Wide, Cross Right, Front Tele
    h, w = cam_frames.shape[1], cam_frames.shape[2]
    new_h, new_w = HEIGHT // 2, WIDTH // 2

    resized = []
    for i in range(4):
        frame = cv2.resize(cam_frames[i], (new_w, new_h))
        resized.append(frame)

    # Layout: [Cross Left | Front Wide] / [Front Tele | Cross Right]
    top = np.hstack([resized[0], resized[1]])     # Cross Left, Front Wide
    bottom = np.hstack([resized[3], resized[2]])  # Front Tele, Cross Right

    quad = np.vstack([top, bottom])
    return quad


def draw_trajectory(img_array, pred_traj, gt_traj=None, scale=25):
    """Draw predicted and ground truth trajectories on the image."""
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)

    # Center point for trajectory visualization (bottom center)
    cx, cy = WIDTH // 2 + 300, HEIGHT - 180

    try:
        font_small = ImageFont.truetype(FONT_PATH, 14)
    except:
        font_small = ImageFont.load_default()

    # Draw background
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(
        [cx - 200, cy - 160, cx + 200, cy + 70],
        radius=10,
        fill=(0, 0, 0, 180)
    )
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay)
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)

    # Draw coordinate grid
    for i in range(-3, 4):
        x = cx + i * scale * 2
        draw.line([(x, cy - 150), (x, cy + 60)], fill=(40, 40, 40), width=1)
    for i in range(-4, 3):
        y = cy + i * scale
        draw.line([(cx - 190, y), (cx + 190, y)], fill=(40, 40, 40), width=1)

    # Draw ego vehicle (small car icon at origin)
    ego_points = [(cx, cy - 12), (cx - 8, cy + 6), (cx + 8, cy + 6)]
    draw.polygon(ego_points, fill=(100, 180, 255))

    # Draw ground truth trajectory (green)
    if gt_traj is not None and len(gt_traj) > 0:
        gt_points = []
        for point in gt_traj[::3]:  # Sample every 3rd point
            px = cx - point[1] * scale
            py = cy - point[0] * scale
            if abs(px - cx) < 190 and abs(py - cy) < 150:
                gt_points.append((px, py))

        if len(gt_points) > 1:
            for i in range(len(gt_points) - 1):
                draw.line([gt_points[i], gt_points[i+1]], fill=(0, 200, 0), width=3)

    # Draw predicted trajectory (yellow/orange)
    if pred_traj is not None and len(pred_traj) > 0:
        pred_points = []
        for point in pred_traj[::3]:
            px = cx - point[1] * scale
            py = cy - point[0] * scale
            if abs(px - cx) < 190 and abs(py - cy) < 150:
                pred_points.append((px, py))

        if len(pred_points) > 1:
            for i in range(len(pred_points) - 1):
                draw.line([pred_points[i], pred_points[i+1]], fill=(255, 180, 0), width=3)
            # Endpoint marker
            if pred_points:
                draw.ellipse([pred_points[-1][0]-5, pred_points[-1][1]-5,
                             pred_points[-1][0]+5, pred_points[-1][1]+5], fill=(255, 100, 0))

    # Title and legend
    draw.text((cx - 190, cy - 155), "Trajectory Prediction (6.4s)", font=font_small, fill=(200, 200, 200))
    draw.rectangle([cx + 80, cy - 155, cx + 95, cy - 145], fill=(0, 200, 0))
    draw.text((cx + 100, cy - 155), "GT", font=font_small, fill=(200, 200, 200))
    draw.rectangle([cx + 130, cy - 155, cx + 145, cy - 145], fill=(255, 180, 0))
    draw.text((cx + 150, cy - 155), "Pred", font=font_small, fill=(200, 200, 200))

    return np.array(img)


def add_overlay(frame, coc_text, ade=None, pred_traj=None, gt_traj=None, scenario_title=""):
    """Add chain-of-causation text overlay, trajectory, and watermark."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(FONT_PATH, 20)
        font_small = ImageFont.truetype(FONT_PATH, 16)
        font_title = ImageFont.truetype(FONT_PATH, 28)
        font_watermark = ImageFont.truetype(FONT_PATH, 42)
    except:
        font = ImageFont.load_default()
        font_small = font
        font_title = font
        font_watermark = font

    # Camera labels with background
    labels_pos = [
        ("Cross Left", 20, 10),
        ("Front Wide", WIDTH // 2 + 20, 10),
        ("Front Tele", 20, HEIGHT // 2 + 10),
        ("Cross Right", WIDTH // 2 + 20, HEIGHT // 2 + 10),
    ]
    for label, x, y in labels_pos:
        bbox = draw.textbbox((x, y), label, font=font)
        draw.rectangle([bbox[0]-4, bbox[1]-2, bbox[2]+4, bbox[3]+2], fill=(0, 0, 0, 200))
        draw.text((x, y), label, font=font, fill=(255, 255, 255))

    # Scenario title (top center)
    if scenario_title:
        bbox = draw.textbbox((0, 0), scenario_title, font=font_title)
        text_width = bbox[2] - bbox[0]
        x = (WIDTH - text_width) // 2
        draw.rectangle([x-15, 5, x + text_width + 15, 45], fill=(0, 0, 0, 200))
        draw.text((x, 10), scenario_title, font=font_title, fill=(100, 200, 255))

    # CoC box (left side)
    box_width = 550
    box_height = 260
    margin = 20
    box_y = HEIGHT - box_height - margin - 80

    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(
        [margin, box_y, margin + box_width, box_y + box_height],
        radius=10,
        fill=(0, 0, 0, 200)
    )
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay)
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)

    # CoC Title
    draw.text((margin + 15, box_y + 10),
              "Chain-of-Causation Reasoning:",
              font=font, fill=(100, 200, 255))

    # Wrap and draw CoC text
    wrapped = textwrap.wrap(coc_text, width=55)[:9]
    y = box_y + 40
    for line in wrapped:
        draw.text((margin + 15, y), line, font=font_small, fill=(255, 255, 255))
        y += 22

    # ADE indicator
    if ade is not None:
        ade_color = (0, 255, 0) if ade < 0.5 else (255, 255, 0) if ade < 1.0 else (255, 100, 100)
        draw.text((margin + 15, box_y + box_height - 30),
                  f"Trajectory ADE: {ade:.2f}m",
                  font=font, fill=ade_color)

    # Draw trajectory visualization
    img_array = np.array(img)
    if pred_traj is not None:
        img_array = draw_trajectory(img_array, pred_traj, gt_traj)

    # Watermark (semi-transparent, bottom right corner)
    img = Image.fromarray(img_array)

    # Create watermark with transparency
    watermark = Image.new('RGBA', img.size, (0, 0, 0, 0))
    wm_draw = ImageDraw.Draw(watermark)
    watermark_text = "hwkim3330"
    bbox = wm_draw.textbbox((0, 0), watermark_text, font=font_watermark)
    wm_width = bbox[2] - bbox[0]
    wm_x = WIDTH - wm_width - 40
    wm_y = HEIGHT - 55
    wm_draw.text((wm_x, wm_y), watermark_text, font=font_watermark, fill=(255, 255, 255, 100))

    img = img.convert('RGBA')
    img = Image.alpha_composite(img, watermark)
    img = img.convert('RGB')

    return np.array(img)


def render_scenario_frames(scenario_data, max_duration_sec=10):
    """Render frames for a single scenario."""
    frames = scenario_data['frames']
    inference_results = scenario_data['inference_results']
    title = scenario_data.get('display_title', scenario_data.get('title', ''))

    num_frames = min(len(frames), max_duration_sec * FPS)

    t_start = scenario_data['t_start_us']

    rendered_frames = []
    current_coc = "Analyzing driving scene..."
    current_ade = None
    current_pred = None
    current_gt = None

    for i in range(num_frames):
        current_t = t_start + i * 100_000

        # Find applicable inference result
        for r in inference_results:
            if r['t0_us'] <= current_t:
                coc = r['coc']
                # Handle different CoC formats (string, list, or numpy array)
                if hasattr(coc, 'tolist'):
                    coc = coc.tolist()
                if isinstance(coc, list):
                    coc = coc[0] if coc else ""
                current_coc = str(coc)
                current_ade = r['ade']
                current_pred = r.get('pred')
                current_gt = r.get('gt')

        cam_frames = frames[i]
        quad = create_quad_view(cam_frames)

        final_frame = add_overlay(
            quad, current_coc, current_ade,
            pred_traj=current_pred, gt_traj=current_gt,
            scenario_title=title
        )
        rendered_frames.append(final_frame)

    return rendered_frames


def generate_tts_audio(coc_texts, output_path):
    """Generate TTS audio for CoC texts using gTTS."""
    try:
        # Convert any non-string CoC texts
        processed_texts = []
        for coc in coc_texts[:8]:
            if hasattr(coc, 'tolist'):
                coc = coc.tolist()
            if isinstance(coc, list):
                coc = coc[0] if coc else ""
            processed_texts.append(str(coc))

        # Combine texts with pauses
        combined_text = " ... ".join(processed_texts)
        # Clean text
        combined_text = combined_text.replace('\n', ' ').strip()
        if len(combined_text) > 1000:
            combined_text = combined_text[:1000]

        print(f"  Generating TTS for {len(combined_text)} characters...")
        tts = gTTS(text=combined_text, lang='en', slow=False)
        tts.save(str(output_path))
        return output_path.exists()
    except Exception as e:
        print(f"  TTS error: {e}")
        return False


def main():
    print("=" * 60)
    print("Final Alpamayo Demo Video")
    print("=" * 60)

    # Try to load new 3-second interval scenarios first, fall back to existing
    scenario_files = []

    # Check for new 3-second interval scenarios
    new_scenarios = [
        ("scenario_3s_a.pkl", "Urban Driving"),
        ("scenario_3s_b.pkl", "Highway Merge"),
        ("scenario_3s_c.pkl", "Intersection"),
    ]

    for filename, title in new_scenarios:
        if (CACHE_DIR / filename).exists():
            scenario_files.append((filename, title))

    # Fall back to existing scenarios if new ones don't exist
    if not scenario_files:
        scenario_files = [
            ("continuous_urban_drive.pkl", "Urban Driving"),
            ("continuous_scenario_b.pkl", "Following Vehicle"),
            ("continuous_scenario_c.pkl", "Intersection"),
        ]

    scenarios = []
    all_coc_texts = []

    for filename, display_title in scenario_files:
        pkl_path = CACHE_DIR / filename
        if pkl_path.exists():
            print(f"Loading {filename}...")
            data = load_scenario(pkl_path)
            data['display_title'] = display_title
            scenarios.append(data)

            for r in data['inference_results']:
                all_coc_texts.append(r['coc'])

            print(f"  - {len(data['frames'])} frames, {len(data['inference_results'])} inferences")

    if not scenarios:
        print("No scenarios found!")
        return

    all_frames = []

    # Render each scenario (no intro/outro as requested)
    for scenario in scenarios:
        print(f"\nRendering {scenario['display_title']}...")
        scenario_frames = render_scenario_frames(scenario, max_duration_sec=10)
        all_frames.extend(scenario_frames)
        print(f"  Added {len(scenario_frames)} frames")

    # Write video
    output_path = OUTPUT_DIR / "alpamayo_final_temp.mp4"
    print(f"\nWriting video: {len(all_frames)} frames ({len(all_frames) / FPS:.1f} sec)")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (WIDTH, HEIGHT))

    for frame in all_frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)

    writer.release()

    # Convert to H.264
    video_only_path = OUTPUT_DIR / "alpamayo_final_v2.mp4"
    print("Converting to H.264...")
    subprocess.run([
        'ffmpeg', '-y', '-i', str(output_path),
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '22',
        '-pix_fmt', 'yuv420p',
        str(video_only_path)
    ], capture_output=True)

    if video_only_path.exists():
        size_mb = video_only_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ Video (no audio): {video_only_path}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Duration: {len(all_frames) / FPS:.1f} seconds")
        output_path.unlink()

    # Generate TTS
    print("\nGenerating TTS audio...")
    tts_path = OUTPUT_DIR / "alpamayo_tts.mp3"
    if generate_tts_audio(all_coc_texts, tts_path):
        print(f"  TTS audio: {tts_path}")

        # Merge audio with video
        final_with_audio = OUTPUT_DIR / "alpamayo_final_v2_tts.mp4"
        print("  Merging video and audio...")
        result = subprocess.run([
            'ffmpeg', '-y',
            '-i', str(video_only_path),
            '-i', str(tts_path),
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k',
            '-shortest',
            str(final_with_audio)
        ], capture_output=True)

        if final_with_audio.exists():
            size_mb = final_with_audio.stat().st_size / (1024 * 1024)
            print(f"\n✓ Video with TTS: {final_with_audio}")
            print(f"  Size: {size_mb:.1f} MB")
    else:
        print("  TTS generation failed")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
