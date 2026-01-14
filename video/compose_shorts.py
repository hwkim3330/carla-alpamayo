#!/usr/bin/env python3
"""
Compose YouTube Shorts version (9:16 vertical) of Alpamayo demo.
"""

import pickle
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap

# Paths
CACHE_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/cache")
OUTPUT_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Video settings for Shorts (9:16)
WIDTH = 1080
HEIGHT = 1920
FPS = 10
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_scenario(pkl_path):
    """Load cached scenario data."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def create_text_frame(text, size=(WIDTH, HEIGHT), bg_color=(20, 20, 30), text_color=(255, 255, 255)):
    """Create a frame with centered text."""
    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)

    try:
        font_large = ImageFont.truetype(FONT_PATH, 56)
        font_small = ImageFont.truetype(FONT_PATH, 28)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    if isinstance(text, str):
        lines = text.split('\n')
    else:
        lines = text

    total_height = len(lines) * 60
    y_start = (size[1] - total_height) // 2

    for i, line in enumerate(lines):
        font = font_large if i == 0 else font_small
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (size[0] - text_width) // 2
        y = y_start + i * 60
        draw.text((x, y), line, font=font, fill=text_color)

    return np.array(img)


def create_intro_frames(duration_sec=3):
    """Create intro frames."""
    frames = []

    title_frame = create_text_frame([
        "Alpamayo-R1",
        "",
        "Vision-Language-Action",
        "Model for",
        "Autonomous Driving",
        "",
        "Real-time Reasoning"
    ])

    for i in range(FPS):
        alpha = i / FPS
        frame = (title_frame * alpha).astype(np.uint8)
        frames.append(frame)

    for _ in range(int((duration_sec - 2) * FPS)):
        frames.append(title_frame)

    for i in range(FPS):
        alpha = 1 - i / FPS
        frame = (title_frame * alpha).astype(np.uint8)
        frames.append(frame)

    return frames


def create_outro_frames(duration_sec=2):
    """Create outro frames."""
    frames = []

    outro_frame = create_text_frame([
        "Alpamayo-R1-10B",
        "",
        "github.com/",
        "NVlabs/alpamayo",
        "",
        "NVIDIA Research"
    ])

    for i in range(int(FPS * 0.5)):
        alpha = i / (FPS * 0.5)
        frame = (outro_frame * alpha).astype(np.uint8)
        frames.append(frame)

    for _ in range(int((duration_sec - 1) * FPS)):
        frames.append(outro_frame)

    for i in range(int(FPS * 0.5)):
        alpha = 1 - i / (FPS * 0.5)
        frame = (outro_frame * alpha).astype(np.uint8)
        frames.append(frame)

    return frames


def create_vertical_view(cam_frames):
    """Create vertical stacked view - Front on top, with side cams below."""
    # cam_frames: (4, H, W, 3) - Left, Front, Right, Rear
    # For shorts: Front cam large on top, Left/Right small below

    front = cam_frames[1]  # Front camera

    # Scale front to fill width
    front_h = int(1080 * (HEIGHT * 0.5 / 1920))
    front_resized = cv2.resize(front, (WIDTH, front_h))

    # Left and Right cameras below
    side_w = WIDTH // 2
    side_h = int(1080 * (side_w / 1920))
    left_resized = cv2.resize(cam_frames[0], (side_w, side_h))
    right_resized = cv2.resize(cam_frames[2], (side_w, side_h))

    # Create canvas
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Place front camera in upper portion
    y_offset = 100
    canvas[y_offset:y_offset + front_h, :] = front_resized

    # Place side cameras
    side_y = y_offset + front_h + 20
    canvas[side_y:side_y + side_h, :side_w] = left_resized
    canvas[side_y:side_y + side_h, side_w:] = right_resized

    return canvas


def add_coc_overlay_vertical(frame, coc_text, ade=None, scenario_title=""):
    """Add chain-of-causation text overlay to vertical frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(FONT_PATH, 24)
        font_title = ImageFont.truetype(FONT_PATH, 32)
        font_small = ImageFont.truetype(FONT_PATH, 18)
    except:
        font = ImageFont.load_default()
        font_title = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Title at top
    if scenario_title:
        bbox = draw.textbbox((0, 0), scenario_title, font=font_title)
        text_width = bbox[2] - bbox[0]
        x = (WIDTH - text_width) // 2
        draw.text((x, 30), scenario_title, font=font_title, fill=(100, 200, 255))

    # Camera labels
    draw.text((20, 120), "Front Camera", font=font_small, fill=(255, 255, 255))
    front_h = int(1080 * (HEIGHT * 0.5 / 1920))
    draw.text((20, 120 + front_h + 40), "Left", font=font_small, fill=(255, 255, 255))
    draw.text((WIDTH // 2 + 20, 120 + front_h + 40), "Right", font=font_small, fill=(255, 255, 255))

    # CoC box at bottom
    box_margin = 20
    box_height = 400
    box_y = HEIGHT - box_height - box_margin

    # Semi-transparent background
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [box_margin, box_y, WIDTH - box_margin, HEIGHT - box_margin],
        fill=(0, 0, 0, 200)
    )
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay)
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)

    # CoC title
    draw.text((box_margin + 15, box_y + 10),
              "Chain-of-Causation:",
              font=font, fill=(100, 200, 255))

    # Wrapped text
    wrapped = textwrap.wrap(coc_text, width=40)[:12]
    y = box_y + 45
    for line in wrapped:
        draw.text((box_margin + 15, y), line, font=font_small, fill=(255, 255, 255))
        y += 26

    # ADE
    if ade is not None:
        ade_color = (0, 255, 0) if ade < 0.5 else (255, 255, 0) if ade < 1.0 else (255, 100, 100)
        draw.text((box_margin + 15, HEIGHT - box_margin - 35),
                  f"Trajectory ADE: {ade:.2f}m",
                  font=font, fill=ade_color)

    return np.array(img)


def render_scenario_frames_vertical(scenario_data, max_duration_sec=8):
    """Render vertical frames for a single scenario."""
    frames = scenario_data['frames']
    inference_results = scenario_data['inference_results']
    title = scenario_data.get('display_title', '')

    num_frames = min(len(frames), max_duration_sec * FPS)

    result_times = [r['t0_us'] for r in inference_results]
    t_start = scenario_data['t_start_us']

    rendered_frames = []
    current_coc = "Initializing reasoning..."
    current_ade = None

    for i in range(num_frames):
        current_t = t_start + i * 100_000

        for j, r in enumerate(inference_results):
            if r['t0_us'] <= current_t:
                current_coc = r['coc']
                current_ade = r['ade']

        cam_frames = frames[i]
        vertical = create_vertical_view(cam_frames)
        final_frame = add_coc_overlay_vertical(vertical, current_coc, current_ade, title)
        rendered_frames.append(final_frame)

    return rendered_frames


def main():
    print("=" * 60)
    print("Composing YouTube Shorts (9:16) Version")
    print("=" * 60)

    # Load scenarios
    scenarios = []
    scenario_files = [
        ("continuous_urban_drive.pkl", "Urban Driving"),
        ("continuous_scenario_b.pkl", "Following Vehicle"),
        ("continuous_scenario_c.pkl", "Intersection"),
    ]

    for filename, display_title in scenario_files:
        pkl_path = CACHE_DIR / filename
        if pkl_path.exists():
            print(f"Loading {filename}...")
            data = load_scenario(pkl_path)
            data['display_title'] = display_title
            scenarios.append(data)

    if not scenarios:
        print("No scenarios found!")
        return

    all_frames = []

    # Intro
    print("\nCreating intro...")
    intro_frames = create_intro_frames(duration_sec=3)
    all_frames.extend(intro_frames)

    # Each scenario (shorter for shorts format - ~8 sec each)
    for i, scenario in enumerate(scenarios):
        print(f"Processing {scenario['display_title']}...")
        scenario_frames = render_scenario_frames_vertical(scenario, max_duration_sec=8)
        all_frames.extend(scenario_frames)

    # Outro
    print("Creating outro...")
    outro_frames = create_outro_frames(duration_sec=2)
    all_frames.extend(outro_frames)

    # Write video
    output_path = OUTPUT_DIR / "alpamayo_shorts_temp.mp4"
    print(f"\nWriting video: {len(all_frames)} frames ({len(all_frames) / FPS:.1f} sec)")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (WIDTH, HEIGHT))

    for frame in all_frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)

    writer.release()

    # Convert to H.264
    final_path = OUTPUT_DIR / "alpamayo_showcase_shorts.mp4"
    print("Converting to H.264...")
    import subprocess
    subprocess.run([
        'ffmpeg', '-y', '-i', str(output_path),
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
        '-pix_fmt', 'yuv420p',
        str(final_path)
    ], capture_output=True)

    if final_path.exists():
        size_mb = final_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ Final shorts video: {final_path}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Duration: {len(all_frames) / FPS:.1f} seconds")
        output_path.unlink()

    print("\nDone!")


if __name__ == "__main__":
    main()
