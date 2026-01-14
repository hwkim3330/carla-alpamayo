#!/usr/bin/env python3
"""
Generate continuous driving scenarios with 3-second inference intervals.
Uses the same inference pattern as test_inference.py
"""

import sys
sys.path.insert(0, '/mnt/data/lfm_agi/alpamayo_code/src')

import pickle
import numpy as np
from pathlib import Path
import torch

# Paths
OUTPUT_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/cache")
OUTPUT_DIR.mkdir(exist_ok=True)

# Settings
VIDEO_DURATION = 12  # seconds
INFERENCE_INTERVAL = 3  # seconds (user requested)
FPS = 10


def generate_scenario(clip_id, name, title, avdi=None, model=None, processor=None):
    """Generate a continuous driving scenario with 3-second inference intervals."""
    import physical_ai_av
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper

    print(f"\n{'='*60}")
    print(f"Generating: {title}")
    print(f"{'='*60}")

    # Initialize dataset interface
    if avdi is None:
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    # Get clip time range from egomotion
    egomotion = avdi.get_clip_feature(clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True)
    t_min, t_max = egomotion.time_range

    # Start time: 3 seconds into clip (need history)
    t_start_us = t_min + 3_000_000
    t_end_us = t_start_us + VIDEO_DURATION * 1_000_000

    if t_end_us > t_max - 2_000_000:
        t_end_us = t_max - 2_000_000
        t_start_us = t_end_us - VIDEO_DURATION * 1_000_000

    print(f"Time range: {t_start_us/1e6:.1f}s - {t_end_us/1e6:.1f}s")

    # Load all frames at 10Hz
    all_timestamps = np.arange(t_start_us, t_end_us, 100_000, dtype=np.int64)
    print(f"Loading {len(all_timestamps)} frames...")

    # Get camera features
    camera_features = [
        avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
    ]

    # Load frames for each camera
    all_cam_frames = []
    for cam_feature in camera_features:
        camera = avdi.get_clip_feature(clip_id, cam_feature, maybe_stream=True)
        frames, _ = camera.decode_images_from_timestamps(all_timestamps)
        all_cam_frames.append(frames)

    # Stack: (T, 4, H, W, 3)
    all_cam_frames = np.stack(all_cam_frames, axis=1)
    print(f"Frames shape: {all_cam_frames.shape}")

    # Initialize model if not provided
    if model is None:
        print("Loading Alpamayo-R1 model...")
        model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
        processor = helper.get_processor(model.tokenizer)

    # Run inference at 3-second intervals
    inference_times = np.arange(
        t_start_us + INFERENCE_INTERVAL * 1_000_000,
        t_end_us - 1_000_000,
        INFERENCE_INTERVAL * 1_000_000
    ).astype(np.int64)

    print(f"Running {len(inference_times)} inferences at {INFERENCE_INTERVAL}-second intervals...")

    inference_results = []
    for i, t0_us in enumerate(inference_times):
        print(f"  Inference {i+1}/{len(inference_times)} at t={t0_us/1e6:.1f}s...")

        # Load sample
        data = load_physical_aiavdataset(
            clip_id=clip_id,
            t0_us=int(t0_us),
            avdi=avdi,
            num_frames=4
        )

        # Prepare inputs
        messages = helper.create_message(data["image_frames"].flatten(0, 1))
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }
        model_inputs = helper.to_device(model_inputs, "cuda")

        # Run inference
        torch.cuda.manual_seed_all(42 + i)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                max_generation_length=256,
                return_extra=True,
            )

        # Extract results
        pred_traj = pred_xyz.cpu().numpy()[0, 0, 0]  # (64, 3)
        gt_traj = data['ego_future_xyz'].cpu().numpy().squeeze()  # (64, 3)

        # Calculate ADE
        ade = np.mean(np.linalg.norm(pred_traj[:, :2] - gt_traj[:, :2], axis=1))

        # Get CoC text
        coc_text = extra['cot'][0][0] if extra['cot'] else "No reasoning available"

        inference_results.append({
            't0_us': int(t0_us),
            'coc': coc_text,
            'pred': pred_traj,
            'gt': gt_traj,
            'ade': float(ade)
        })

        print(f"    ADE: {ade:.2f}m")
        print(f"    CoC: {coc_text[:100]}...")

    # Save
    output_data = {
        'name': name,
        'title': title,
        'clip_id': clip_id,
        'frames': all_cam_frames,
        'timestamps': all_timestamps,
        't_start_us': int(t_start_us),
        'inference_results': inference_results
    }

    output_path = OUTPUT_DIR / f"{name}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"Saved: {output_path}")
    return output_path, model, processor, avdi


def main():
    import physical_ai_av
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper

    # Initialize once
    print("Initializing...")
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    # Get available clips from clip_index
    clips = avdi.clip_index.index.tolist()
    print(f"Found {len(clips)} clips")

    # Load model once
    print("Loading Alpamayo-R1 model...")
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    print("Model loaded!")

    # Scenarios to generate
    scenarios = [
        {
            'clip_id': clips[0],
            'name': 'scenario_3s_a',
            'title': 'Urban Driving'
        },
        {
            'clip_id': clips[min(5, len(clips)-1)],
            'name': 'scenario_3s_b',
            'title': 'Highway Merge'
        },
        {
            'clip_id': clips[min(10, len(clips)-1)],
            'name': 'scenario_3s_c',
            'title': 'Intersection'
        },
    ]

    for s in scenarios:
        try:
            generate_scenario(
                clip_id=s['clip_id'],
                name=s['name'],
                title=s['title'],
                avdi=avdi,
                model=model,
                processor=processor
            )
        except Exception as e:
            print(f"Error generating {s['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*60)
    print("All scenarios generated!")
    print("="*60)


if __name__ == "__main__":
    main()
