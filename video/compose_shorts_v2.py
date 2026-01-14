#!/usr/bin/env python3
"""
Alpamayo YouTube Shorts (9:16) version with:
- Correct camera layout for vertical format
- hwkim3330 watermark
- Trajectory visualization
"""

import pickle
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap
import subprocess

CACHE_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/cache")
OUTPUT_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/output")

WIDTH = 1080
HEIGHT = 1920
FPS = 10
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_scenario(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def create_vertical_view(cam_frames):
    """Create vertical layout: Front Wide on top, Cross Left/Right below."""
    # cam_frames: (4, H, W, 3) - Cross Left, Front Wide, Cross Right, Front Tele
    front_wide = cam_frames[1]  # Main view

    # Scale front to full width
    front_h = int(1080 * (WIDTH / 1920))
    front_resized = cv2.resize(front_wide, (WIDTH, front_h))

    # Left and Right below (side by side)
    side_w = WIDTH // 2
    side_h = int(1080 * (side_w / 1920))
    left_resized = cv2.resize(cam_frames[0], (side_w, side_h))
    right_resized = cv2.resize(cam_frames[2], (side_w, side_h))
    sides = np.hstack([left_resized, right_resized])

    # Create canvas
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Place cameras
    y_offset = 100
    canvas[y_offset:y_offset + front_h, :] = front_resized
    side_y = y_offset + front_h + 20
    canvas[side_y:side_y + side_h, :] = sides

    return canvas


def draw_trajectory_vertical(img_array, pred_traj, gt_traj=None, scale=20):
    """Draw trajectories for vertical format."""
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)

    cx, cy = WIDTH // 2, HEIGHT - 550

    try:
        font_small = ImageFont.truetype(FONT_PATH, 12)
    except:
        font_small = ImageFont.load_default()

    # Background
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(
        [cx - 150, cy - 120, cx + 150, cy + 50],
        radius=8,
        fill=(0, 0, 0, 180)
    )
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay)
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)

    # Ego vehicle
    ego_points = [(cx, cy - 10), (cx - 6, cy + 5), (cx + 6, cy + 5)]
    draw.polygon(ego_points, fill=(100, 180, 255))

    # GT trajectory (green)
    if gt_traj is not None:
        gt_points = []
        for point in gt_traj[::4]:
            px = cx - point[1] * scale
            py = cy - point[0] * scale
            if abs(px - cx) < 140 and abs(py - cy) < 110:
                gt_points.append((px, py))
        if len(gt_points) > 1:
            for i in range(len(gt_points) - 1):
                draw.line([gt_points[i], gt_points[i+1]], fill=(0, 200, 0), width=2)

    # Predicted trajectory (orange)
    if pred_traj is not None:
        pred_points = []
        for point in pred_traj[::4]:
            px = cx - point[1] * scale
            py = cy - point[0] * scale
            if abs(px - cx) < 140 and abs(py - cy) < 110:
                pred_points.append((px, py))
        if len(pred_points) > 1:
            for i in range(len(pred_points) - 1):
                draw.line([pred_points[i], pred_points[i+1]], fill=(255, 180, 0), width=2)

    # Legend
    draw.text((cx - 140, cy - 115), "Trajectory", font=font_small, fill=(200, 200, 200))

    return np.array(img)


def add_overlay_vertical(frame, coc_text, ade=None, pred_traj=None, gt_traj=None, scenario_title=""):
    """Add overlays for vertical format."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(FONT_PATH, 22)
        font_small = ImageFont.truetype(FONT_PATH, 16)
        font_title = ImageFont.truetype(FONT_PATH, 28)
        font_watermark = ImageFont.truetype(FONT_PATH, 36)
    except:
        font = font_small = font_title = font_watermark = ImageFont.load_default()

    # Title (top)
    if scenario_title:
        bbox = draw.textbbox((0, 0), scenario_title, font=font_title)
        text_width = bbox[2] - bbox[0]
        x = (WIDTH - text_width) // 2
        draw.rectangle([x-10, 30, x + text_width + 10, 70], fill=(0, 0, 0, 200))
        draw.text((x, 35), scenario_title, font=font_title, fill=(100, 200, 255))

    # Camera labels
    front_h = int(1080 * (WIDTH / 1920))
    draw.text((15, 105), "Front Wide", font=font_small, fill=(255, 255, 255))
    side_y = 100 + front_h + 25
    draw.text((15, side_y), "Cross Left", font=font_small, fill=(255, 255, 255))
    draw.text((WIDTH // 2 + 15, side_y), "Cross Right", font=font_small, fill=(255, 255, 255))

    # CoC box (bottom)
    box_margin = 15
    box_height = 380
    box_y = HEIGHT - box_height - box_margin

    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(
        [box_margin, box_y, WIDTH - box_margin, HEIGHT - box_margin],
        radius=10,
        fill=(0, 0, 0, 200)
    )
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay)
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)

    # CoC title and text
    draw.text((box_margin + 15, box_y + 15),
              "Chain-of-Causation:",
              font=font, fill=(100, 200, 255))

    wrapped = textwrap.wrap(coc_text, width=38)[:10]
    y = box_y + 50
    for line in wrapped:
        draw.text((box_margin + 15, y), line, font=font_small, fill=(255, 255, 255))
        y += 24

    # ADE
    if ade is not None:
        ade_color = (0, 255, 0) if ade < 0.5 else (255, 255, 0) if ade < 1.0 else (255, 100, 100)
        draw.text((box_margin + 15, HEIGHT - box_margin - 35),
                  f"ADE: {ade:.2f}m", font=font, fill=ade_color)

    # Draw trajectory
    img_array = np.array(img)
    if pred_traj is not None:
        img_array = draw_trajectory_vertical(img_array, pred_traj, gt_traj)

    # Watermark
    img = Image.fromarray(img_array)
    watermark = Image.new('RGBA', img.size, (0, 0, 0, 0))
    wm_draw = ImageDraw.Draw(watermark)
    wm_text = "hwkim3330"
    bbox = wm_draw.textbbox((0, 0), wm_text, font=font_watermark)
    wm_width = bbox[2] - bbox[0]
    wm_x = WIDTH - wm_width - 20
    wm_y = HEIGHT - box_height - 60
    wm_draw.text((wm_x, wm_y), wm_text, font=font_watermark, fill=(255, 255, 255, 80))
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, watermark)
    img = img.convert('RGB')

    return np.array(img)


def render_scenario_vertical(scenario_data, max_duration_sec=8):
    """Render vertical frames for a scenario."""
    frames = scenario_data['frames']
    inference_results = scenario_data['inference_results']
    title = scenario_data.get('display_title', scenario_data.get('title', ''))

    num_frames = min(len(frames), max_duration_sec * FPS)
    t_start = scenario_data['t_start_us']

    rendered_frames = []
    current_coc = "Analyzing..."
    current_ade = None
    current_pred = None
    current_gt = None

    for i in range(num_frames):
        current_t = t_start + i * 100_000

        for r in inference_results:
            if r['t0_us'] <= current_t:
                coc = r['coc']
                if hasattr(coc, 'tolist'):
                    coc = coc.tolist()
                if isinstance(coc, list):
                    coc = coc[0] if coc else ""
                current_coc = str(coc)
                current_ade = r['ade']
                current_pred = r.get('pred')
                current_gt = r.get('gt')

        cam_frames = frames[i]
        vertical = create_vertical_view(cam_frames)
        final_frame = add_overlay_vertical(
            vertical, current_coc, current_ade,
            pred_traj=current_pred, gt_traj=current_gt,
            scenario_title=title
        )
        rendered_frames.append(final_frame)

    return rendered_frames


def main():
    print("=" * 50)
    print("Alpamayo YouTube Shorts (9:16)")
    print("=" * 50)

    # Try new 3-second interval scenarios first
    scenario_files = [
        ("scenario_3s_a.pkl", "Urban Driving"),
        ("scenario_3s_b.pkl", "Highway Merge"),
        ("scenario_3s_c.pkl", "Intersection"),
    ]

    # Fall back to old scenarios if new ones don't exist
    if not (CACHE_DIR / scenario_files[0][0]).exists():
        scenario_files = [
            ("continuous_urban_drive.pkl", "Urban Driving"),
            ("continuous_scenario_b.pkl", "Following Vehicle"),
            ("continuous_scenario_c.pkl", "Intersection"),
        ]

    scenarios = []
    for filename, title in scenario_files:
        pkl_path = CACHE_DIR / filename
        if pkl_path.exists():
            print(f"Loading {filename}...")
            data = load_scenario(pkl_path)
            data['display_title'] = title
            scenarios.append(data)

    if not scenarios:
        print("No scenarios found!")
        return

    all_frames = []

    for scenario in scenarios:
        print(f"Rendering {scenario['display_title']}...")
        frames = render_scenario_vertical(scenario, max_duration_sec=8)
        all_frames.extend(frames)
        print(f"  Added {len(frames)} frames")

    # Write video
    output_path = OUTPUT_DIR / "alpamayo_shorts_v2_temp.mp4"
    print(f"\nWriting {len(all_frames)} frames ({len(all_frames)/FPS:.1f}s)...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (WIDTH, HEIGHT))

    for frame in all_frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()

    # Convert
    final_path = OUTPUT_DIR / "alpamayo_shorts_v2.mp4"
    print("Converting...")
    subprocess.run([
        'ffmpeg', '-y', '-i', str(output_path),
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
        '-pix_fmt', 'yuv420p',
        str(final_path)
    ], capture_output=True)

    if final_path.exists():
        size_mb = final_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ {final_path}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Duration: {len(all_frames)/FPS:.1f}s")
        output_path.unlink()

    print("\nDone!")


if __name__ == "__main__":
    main()
