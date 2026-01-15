#!/usr/bin/env python3
"""
Cinematic Alpamayo video - more realistic, less overlay clutter.
Focus on Front Wide camera with minimal UI.
"""

import pickle
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import subprocess

CACHE_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/cache")
OUTPUT_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/output")

WIDTH = 1920
HEIGHT = 1080
FPS = 10
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_scenario(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def add_minimal_overlay(frame, coc_text=None, ade=None):
    """Minimal overlay - just subtitle-style CoC at bottom."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(FONT_PATH, 24)
        font_small = ImageFont.truetype(FONT_PATH, 18)
        font_wm = ImageFont.truetype(FONT_PATH, 32)
    except:
        font = font_small = font_wm = ImageFont.load_default()

    # Subtle CoC text at bottom (subtitle style)
    if coc_text:
        # Semi-transparent bar at bottom
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rectangle([0, HEIGHT-80, WIDTH, HEIGHT], fill=(0, 0, 0, 150))
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img)

        # Truncate long text
        if len(coc_text) > 100:
            coc_text = coc_text[:100] + "..."

        # Center text
        bbox = draw.textbbox((0, 0), coc_text, font=font)
        tw = bbox[2] - bbox[0]
        tx = (WIDTH - tw) // 2
        draw.text((tx, HEIGHT - 60), coc_text, font=font, fill=(255, 255, 255))

        # ADE in corner
        if ade is not None:
            ade_color = (100, 255, 100) if ade < 0.5 else (255, 255, 100) if ade < 1.0 else (255, 150, 150)
            draw.text((WIDTH - 150, HEIGHT - 60), f"ADE: {ade:.1f}m", font=font_small, fill=ade_color)

    # Watermark (very subtle)
    wm = Image.new('RGBA', img.size, (0, 0, 0, 0))
    wd = ImageDraw.Draw(wm)
    wd.text((WIDTH - 180, 20), "hwkim3330", font=font_wm, fill=(255, 255, 255, 60))
    img = Image.alpha_composite(img.convert('RGBA'), wm).convert('RGB')

    return np.array(img)


def render_cinematic(data):
    """Render cinematic video - Front Wide camera only."""
    frames = data['frames']  # (T, 4, H, W, 3)
    results = data['inference_results']
    t_start = data['t_start_us']

    rendered = []
    coc, ade = None, None

    for i in range(len(frames)):
        t = t_start + i * 100_000

        # Update CoC/ADE when new inference available
        for r in results:
            if r['t0_us'] <= t:
                c = r['coc']
                if hasattr(c, 'tolist'): c = c.tolist()
                if isinstance(c, list): c = c[0] if c else ""
                coc = str(c)
                ade = r['ade']

        # Use Front Wide camera (index 1)
        front_wide = frames[i][1]  # (H, W, 3)

        # Resize to 1080p if needed
        if front_wide.shape[:2] != (HEIGHT, WIDTH):
            front_wide = cv2.resize(front_wide, (WIDTH, HEIGHT))

        final = add_minimal_overlay(front_wide, coc, ade)
        rendered.append(final)

    return rendered


def main():
    print("=" * 50)
    print("Cinematic Alpamayo Video")
    print("=" * 50)

    # Load all scenarios
    scenario_files = [
        "scenario_3s_a.pkl", "scenario_3s_b.pkl", "scenario_3s_c.pkl",
        "scenario_5s_d.pkl", "scenario_5s_e.pkl", "scenario_5s_f.pkl",
    ]

    all_frames = []

    for fn in scenario_files:
        path = CACHE_DIR / fn
        if path.exists():
            print(f"Loading {fn}...")
            data = load_scenario(path)
            frames = render_cinematic(data)
            all_frames.extend(frames)
            print(f"  Added {len(frames)} frames")

    if not all_frames:
        print("No data!")
        return

    # Write video
    tmp = OUTPUT_DIR / "alpamayo_cinematic_temp.mp4"
    print(f"\nWriting {len(all_frames)} frames ({len(all_frames)/FPS:.0f}s)...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(tmp), fourcc, FPS, (WIDTH, HEIGHT))
    for f in all_frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

    # Convert to H.264
    final = OUTPUT_DIR / "alpamayo_cinematic.mp4"
    print("Converting...")
    subprocess.run(['ffmpeg', '-y', '-i', str(tmp),
                    '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
                    '-pix_fmt', 'yuv420p', str(final)], capture_output=True)

    if final.exists():
        mb = final.stat().st_size / (1024*1024)
        print(f"\n✓ {final}")
        print(f"  {mb:.1f} MB, {len(all_frames)/FPS:.0f}s")
        tmp.unlink()

    print("\nDone!")


if __name__ == "__main__":
    main()
