#!/usr/bin/env python3
"""
Long continuous Alpamayo demo video.
- All available frames from each scenario
- Clean layout with trajectory + CoC
- hwkim3330 watermark
- No intro/outro, no TTS
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

WIDTH = 1920
HEIGHT = 1080
FPS = 10
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_scenario(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def create_quad_view(cam_frames):
    """2x2 grid: [Cross Left | Front Wide] / [Front Tele | Cross Right]"""
    new_h, new_w = HEIGHT // 2, WIDTH // 2
    resized = [cv2.resize(cam_frames[i], (new_w, new_h)) for i in range(4)]
    top = np.hstack([resized[0], resized[1]])
    bottom = np.hstack([resized[3], resized[2]])
    return np.vstack([top, bottom])


def draw_trajectory(img, pred_traj, gt_traj, scale=25):
    """Draw trajectories in bottom-right area."""
    draw = ImageDraw.Draw(img)
    cx, cy = WIDTH - 250, HEIGHT - 200

    try:
        font = ImageFont.truetype(FONT_PATH, 14)
    except:
        font = ImageFont.load_default()

    # Background
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle([cx-180, cy-150, cx+180, cy+80], radius=8, fill=(0, 0, 0, 180))
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(img)

    # Grid
    for i in range(-3, 4):
        draw.line([(cx + i*40, cy-140), (cx + i*40, cy+70)], fill=(40, 40, 40))
    for i in range(-3, 3):
        draw.line([(cx-170, cy + i*40), (cx+170, cy + i*40)], fill=(40, 40, 40))

    # Ego vehicle
    draw.polygon([(cx, cy-10), (cx-7, cy+5), (cx+7, cy+5)], fill=(100, 180, 255))

    # GT (green)
    if gt_traj is not None:
        pts = [(cx - p[1]*scale, cy - p[0]*scale) for p in gt_traj[::3]
               if abs(p[1]*scale) < 170 and abs(p[0]*scale) < 140]
        for i in range(len(pts)-1):
            draw.line([pts[i], pts[i+1]], fill=(0, 200, 0), width=3)

    # Pred (orange)
    if pred_traj is not None:
        pts = [(cx - p[1]*scale, cy - p[0]*scale) for p in pred_traj[::3]
               if abs(p[1]*scale) < 170 and abs(p[0]*scale) < 140]
        for i in range(len(pts)-1):
            draw.line([pts[i], pts[i+1]], fill=(255, 180, 0), width=3)
        if pts:
            draw.ellipse([pts[-1][0]-4, pts[-1][1]-4, pts[-1][0]+4, pts[-1][1]+4], fill=(255, 100, 0))

    # Legend
    draw.text((cx-170, cy-145), "Trajectory (6.4s)", font=font, fill=(180, 180, 180))
    draw.rectangle([cx+60, cy-145, cx+75, cy-135], fill=(0, 200, 0))
    draw.text((cx+80, cy-145), "GT", font=font, fill=(180, 180, 180))
    draw.rectangle([cx+110, cy-145, cx+125, cy-135], fill=(255, 180, 0))
    draw.text((cx+130, cy-145), "Pred", font=font, fill=(180, 180, 180))

    return img


def add_overlay(frame, coc_text, ade, pred_traj, gt_traj, title):
    """Add all overlays."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(FONT_PATH, 20)
        font_sm = ImageFont.truetype(FONT_PATH, 16)
        font_title = ImageFont.truetype(FONT_PATH, 26)
        font_wm = ImageFont.truetype(FONT_PATH, 40)
    except:
        font = font_sm = font_title = font_wm = ImageFont.load_default()

    # Camera labels
    labels = [("Cross Left", 15, 8), ("Front Wide", WIDTH//2+15, 8),
              ("Front Tele", 15, HEIGHT//2+8), ("Cross Right", WIDTH//2+15, HEIGHT//2+8)]
    for lbl, x, y in labels:
        bbox = draw.textbbox((x, y), lbl, font=font)
        draw.rectangle([bbox[0]-3, bbox[1]-2, bbox[2]+3, bbox[3]+2], fill=(0, 0, 0, 200))
        draw.text((x, y), lbl, font=font, fill=(255, 255, 255))

    # Title
    bbox = draw.textbbox((0, 0), title, font=font_title)
    tw = bbox[2] - bbox[0]
    tx = (WIDTH - tw) // 2
    draw.rectangle([tx-12, 5, tx+tw+12, 40], fill=(0, 0, 0, 200))
    draw.text((tx, 8), title, font=font_title, fill=(100, 200, 255))

    # CoC box (left side)
    box_w, box_h = 520, 240
    margin = 15
    box_y = HEIGHT - box_h - margin - 60

    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle([margin, box_y, margin+box_w, box_y+box_h], radius=8, fill=(0, 0, 0, 190))
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(img)

    draw.text((margin+12, box_y+8), "Chain-of-Causation:", font=font, fill=(100, 200, 255))

    wrapped = textwrap.wrap(str(coc_text), width=52)[:8]
    y = box_y + 38
    for line in wrapped:
        draw.text((margin+12, y), line, font=font_sm, fill=(255, 255, 255))
        y += 22

    # ADE
    if ade is not None:
        ade_color = (0, 255, 0) if ade < 0.5 else (255, 255, 0) if ade < 1.0 else (255, 100, 100)
        draw.text((margin+12, box_y+box_h-28), f"ADE: {ade:.2f}m", font=font, fill=ade_color)

    # Trajectory
    img = draw_trajectory(img, pred_traj, gt_traj)

    # Watermark
    wm = Image.new('RGBA', img.size, (0, 0, 0, 0))
    wd = ImageDraw.Draw(wm)
    bbox = wd.textbbox((0, 0), "hwkim3330", font=font_wm)
    wmw = bbox[2] - bbox[0]
    wd.text((WIDTH - wmw - 25, HEIGHT - 50), "hwkim3330", font=font_wm, fill=(255, 255, 255, 90))
    img = Image.alpha_composite(img.convert('RGBA'), wm).convert('RGB')

    return np.array(img)


def render_scenario(data):
    """Render all frames from a scenario."""
    frames = data['frames']
    results = data['inference_results']
    title = data.get('display_title', data.get('title', ''))
    t_start = data['t_start_us']

    rendered = []
    coc, ade, pred, gt = "Analyzing...", None, None, None

    for i in range(len(frames)):
        t = t_start + i * 100_000
        for r in results:
            if r['t0_us'] <= t:
                c = r['coc']
                if hasattr(c, 'tolist'): c = c.tolist()
                if isinstance(c, list): c = c[0] if c else ""
                coc, ade = str(c), r['ade']
                pred, gt = r.get('pred'), r.get('gt')

        quad = create_quad_view(frames[i])
        final = add_overlay(quad, coc, ade, pred, gt, title)
        rendered.append(final)

    return rendered


def main():
    print("=" * 50)
    print("Alpamayo Long Demo Video")
    print("=" * 50)

    # Load all available scenarios
    scenario_files = [
        ("scenario_3s_a.pkl", "Urban Driving"),
        ("scenario_3s_b.pkl", "Highway Merge"),
        ("scenario_3s_c.pkl", "Intersection"),
    ]

    scenarios = []
    for fn, title in scenario_files:
        path = CACHE_DIR / fn
        if path.exists():
            print(f"Loading {fn}...")
            d = load_scenario(path)
            d['display_title'] = title
            scenarios.append(d)
            print(f"  {len(d['frames'])} frames")

    if not scenarios:
        print("No scenarios!")
        return

    all_frames = []
    for s in scenarios:
        print(f"Rendering {s['display_title']}...")
        frames = render_scenario(s)
        all_frames.extend(frames)
        print(f"  Added {len(frames)} frames")

    # Write video
    tmp = OUTPUT_DIR / "alpamayo_long_temp.mp4"
    print(f"\nWriting {len(all_frames)} frames ({len(all_frames)/FPS:.1f}s)...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(tmp), fourcc, FPS, (WIDTH, HEIGHT))
    for f in all_frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

    # Convert
    final = OUTPUT_DIR / "alpamayo_long.mp4"
    print("Converting to H.264...")
    subprocess.run(['ffmpeg', '-y', '-i', str(tmp),
                    '-c:v', 'libx264', '-preset', 'medium', '-crf', '22',
                    '-pix_fmt', 'yuv420p', str(final)], capture_output=True)

    if final.exists():
        mb = final.stat().st_size / (1024*1024)
        print(f"\n✓ {final}")
        print(f"  Size: {mb:.1f} MB")
        print(f"  Duration: {len(all_frames)/FPS:.1f}s")
        tmp.unlink()

    print("\nDone!")


if __name__ == "__main__":
    main()
