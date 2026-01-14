#!/usr/bin/env python3
"""
Compose final Alpamayo demo video with:
- Professional intro
- Multiple continuous driving scenarios
- Smooth transitions
- Real-time chain-of-causation overlays
- Professional outro
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

# Video settings
WIDTH = 1920
HEIGHT = 1080
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
        font_large = ImageFont.truetype(FONT_PATH, 72)
        font_small = ImageFont.truetype(FONT_PATH, 36)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    if isinstance(text, str):
        lines = text.split('\n')
    else:
        lines = text

    total_height = len(lines) * 80
    y_start = (size[1] - total_height) // 2

    for i, line in enumerate(lines):
        font = font_large if i == 0 else font_small
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (size[0] - text_width) // 2
        y = y_start + i * 80
        draw.text((x, y), line, font=font, fill=text_color)

    return np.array(img)


def create_intro_frames(duration_sec=4):
    """Create intro frames."""
    frames = []

    # Title card
    title_frame = create_text_frame([
        "Alpamayo-R1",
        "",
        "Vision-Language-Action Model",
        "for Autonomous Driving",
        "",
        "Real-time Chain-of-Causation Reasoning"
    ])

    # Fade in
    for i in range(FPS):
        alpha = i / FPS
        frame = (title_frame * alpha).astype(np.uint8)
        frames.append(frame)

    # Hold
    for _ in range(int((duration_sec - 2) * FPS)):
        frames.append(title_frame)

    # Fade out
    for i in range(FPS):
        alpha = 1 - i / FPS
        frame = (title_frame * alpha).astype(np.uint8)
        frames.append(frame)

    return frames


def create_scenario_title_frames(title, duration_sec=2):
    """Create scenario title card frames."""
    frames = []

    title_frame = create_text_frame([
        title,
        "",
        "Continuous Driving with Real-time Reasoning"
    ], bg_color=(30, 30, 50))

    # Fade in
    for i in range(int(FPS * 0.5)):
        alpha = i / (FPS * 0.5)
        frame = (title_frame * alpha).astype(np.uint8)
        frames.append(frame)

    # Hold
    for _ in range(int((duration_sec - 1) * FPS)):
        frames.append(title_frame)

    # Fade out
    for i in range(int(FPS * 0.5)):
        alpha = 1 - i / (FPS * 0.5)
        frame = (title_frame * alpha).astype(np.uint8)
        frames.append(frame)

    return frames


def create_outro_frames(duration_sec=3):
    """Create outro frames."""
    frames = []

    outro_frame = create_text_frame([
        "Alpamayo-R1-10B",
        "",
        "github.com/NVlabs/alpamayo",
        "",
        "Powered by NuRec Dataset",
        "NVIDIA Research"
    ])

    # Fade in
    for i in range(FPS):
        alpha = i / FPS
        frame = (outro_frame * alpha).astype(np.uint8)
        frames.append(frame)

    # Hold
    for _ in range(int((duration_sec - 2) * FPS)):
        frames.append(outro_frame)

    # Fade out
    for i in range(FPS):
        alpha = 1 - i / FPS
        frame = (outro_frame * alpha).astype(np.uint8)
        frames.append(frame)

    return frames


def create_quad_view(cam_frames):
    """Create 2x2 grid view from 4 camera frames."""
    # cam_frames: (4, H, W, 3) - Left, Front, Right, Rear
    h, w = cam_frames.shape[1], cam_frames.shape[2]

    # Scale to fit half resolution
    new_h, new_w = HEIGHT // 2, WIDTH // 2

    resized = []
    for i in range(4):
        frame = cv2.resize(cam_frames[i], (new_w, new_h))
        resized.append(frame)

    # Arrange: [Front, Left] / [Right, Rear]
    # Better arrangement: [Left, Front] / [Rear, Right]
    top = np.hstack([resized[0], resized[1]])  # Left, Front
    bottom = np.hstack([resized[3], resized[2]])  # Rear, Right

    quad = np.vstack([top, bottom])
    return quad


def add_coc_overlay(frame, coc_text, ade=None):
    """Add chain-of-causation text overlay to frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(FONT_PATH, 20)
        font_small = ImageFont.truetype(FONT_PATH, 16)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Semi-transparent overlay box
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # Box dimensions
    box_width = 600
    box_height = 300
    margin = 20

    # Draw background box
    overlay_draw.rectangle(
        [margin, HEIGHT - box_height - margin, margin + box_width, HEIGHT - margin],
        fill=(0, 0, 0, 180)
    )

    # Convert to RGB for compositing
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay)
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((margin + 10, HEIGHT - box_height - margin + 10),
              "Chain-of-Causation Reasoning:",
              font=font, fill=(100, 200, 255))

    # Wrap and draw CoC text
    wrapped = textwrap.wrap(coc_text, width=60)[:10]  # Limit lines
    y = HEIGHT - box_height - margin + 40
    for line in wrapped:
        draw.text((margin + 10, y), line, font=font_small, fill=(255, 255, 255))
        y += 22

    # ADE indicator
    if ade is not None:
        ade_color = (0, 255, 0) if ade < 0.5 else (255, 255, 0) if ade < 1.0 else (255, 100, 100)
        draw.text((margin + 10, HEIGHT - margin - 30),
                  f"Trajectory ADE: {ade:.2f}m",
                  font=font, fill=ade_color)

    # Camera labels
    labels = [("Left Cam", margin + 10, 10),
              ("Front Cam", WIDTH // 2 + 10, 10),
              ("Rear Cam", margin + 10, HEIGHT // 2 + 10),
              ("Right Cam", WIDTH // 2 + 10, HEIGHT // 2 + 10)]

    for label, x, y in labels:
        draw.text((x, y), label, font=font, fill=(255, 255, 255))

    return np.array(img)


def render_scenario_frames(scenario_data, max_duration_sec=10):
    """Render frames for a single scenario."""
    frames = scenario_data['frames']  # (T, 4, H, W, 3)
    inference_results = scenario_data['inference_results']

    num_frames = min(len(frames), max_duration_sec * FPS)

    # Build time-indexed CoC lookup
    result_times = [r['t0_us'] for r in inference_results]
    t_start = scenario_data['t_start_us']

    rendered_frames = []
    current_coc = "Initializing reasoning..."
    current_ade = None

    for i in range(num_frames):
        # Calculate current timestamp
        current_t = t_start + i * 100_000  # 10Hz = 100ms intervals

        # Find applicable inference result
        for j, r in enumerate(inference_results):
            if r['t0_us'] <= current_t:
                current_coc = r['coc']
                current_ade = r['ade']

        # Create quad view
        cam_frames = frames[i]  # (4, H, W, 3)
        quad = create_quad_view(cam_frames)

        # Add overlay
        final_frame = add_coc_overlay(quad, current_coc, current_ade)
        rendered_frames.append(final_frame)

    return rendered_frames


def create_transition_frames(from_frames, to_frames, duration_sec=0.5):
    """Create crossfade transition between two frame sequences."""
    num_transition = int(duration_sec * FPS)

    # Use last frame of 'from' and first frame of 'to'
    from_frame = from_frames[-1] if from_frames else np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    to_frame = to_frames[0] if to_frames else np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    transition = []
    for i in range(num_transition):
        alpha = i / num_transition
        blended = (from_frame * (1 - alpha) + to_frame * alpha).astype(np.uint8)
        transition.append(blended)

    return transition


def main():
    print("=" * 60)
    print("Composing Final Alpamayo Demo Video")
    print("=" * 60)

    # Load all scenarios
    scenarios = []
    scenario_files = [
        ("continuous_urban_drive.pkl", "Scenario 1: Urban Driving"),
        ("continuous_scenario_b.pkl", "Scenario 2: Following Vehicle"),
        ("continuous_scenario_c.pkl", "Scenario 3: Intersection Navigation"),
    ]

    for filename, display_title in scenario_files:
        pkl_path = CACHE_DIR / filename
        if pkl_path.exists():
            print(f"Loading {filename}...")
            data = load_scenario(pkl_path)
            data['display_title'] = display_title
            scenarios.append(data)
            print(f"  - {data.get('title', 'Unknown')} ({len(data['frames'])} frames)")

    if not scenarios:
        print("No scenarios found!")
        return

    all_frames = []

    # 1. Intro
    print("\nCreating intro...")
    intro_frames = create_intro_frames(duration_sec=4)
    all_frames.extend(intro_frames)
    print(f"  Added {len(intro_frames)} intro frames")

    # 2. Each scenario with title cards and transitions
    for i, scenario in enumerate(scenarios):
        print(f"\nProcessing {scenario['display_title']}...")

        # Scenario title card
        title_frames = create_scenario_title_frames(scenario['display_title'], duration_sec=2)

        # Transition from previous to title
        if all_frames:
            trans = create_transition_frames(all_frames[-10:], title_frames)
            all_frames.extend(trans)

        all_frames.extend(title_frames)
        print(f"  Added {len(title_frames)} title frames")

        # Render scenario
        scenario_frames = render_scenario_frames(scenario, max_duration_sec=10)

        # Transition from title to scenario
        trans = create_transition_frames(title_frames, scenario_frames)
        all_frames.extend(trans)

        all_frames.extend(scenario_frames)
        print(f"  Added {len(scenario_frames)} scenario frames")

    # 3. Outro
    print("\nCreating outro...")
    outro_frames = create_outro_frames(duration_sec=3)

    # Transition to outro
    trans = create_transition_frames(all_frames[-10:], outro_frames)
    all_frames.extend(trans)
    all_frames.extend(outro_frames)
    print(f"  Added {len(outro_frames)} outro frames")

    # 4. Write video
    output_path = OUTPUT_DIR / "alpamayo_showcase.mp4"
    print(f"\nWriting video to {output_path}")
    print(f"Total frames: {len(all_frames)}")
    print(f"Duration: {len(all_frames) / FPS:.1f} seconds")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (WIDTH, HEIGHT))

    for i, frame in enumerate(all_frames):
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)

        if (i + 1) % 100 == 0:
            print(f"  Written {i + 1}/{len(all_frames)} frames...")

    writer.release()

    # Convert to better codec
    final_path = OUTPUT_DIR / "alpamayo_showcase_final.mp4"
    print(f"\nConverting to H.264 codec...")
    import subprocess
    subprocess.run([
        'ffmpeg', '-y', '-i', str(output_path),
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
        '-pix_fmt', 'yuv420p',
        str(final_path)
    ], capture_output=True)

    if final_path.exists():
        size_mb = final_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ Final video: {final_path}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Duration: {len(all_frames) / FPS:.1f} seconds")
        # Remove intermediate file
        output_path.unlink()
    else:
        print(f"\n✓ Video saved: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
