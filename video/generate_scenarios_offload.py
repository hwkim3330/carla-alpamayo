#!/usr/bin/env python3
"""Generate scenarios with CPU offloading to save GPU memory."""

import sys
sys.path.insert(0, '/mnt/data/lfm_agi/alpamayo_code/src')

import pickle
import numpy as np
from pathlib import Path
import torch
import gc

OUTPUT_DIR = Path("/mnt/data/lfm_agi/carla-alpamayo/video/cache")
VIDEO_DURATION = 15
INFERENCE_INTERVAL = 5
FPS = 10


def generate_scenario(clip_id, name, title, avdi, model, processor):
    """Generate scenario with memory-efficient approach."""
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper

    print(f"\n{'='*50}")
    print(f"Generating: {title}")
    print(f"{'='*50}")

    egomotion = avdi.get_clip_feature(clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True)
    t_min, t_max = egomotion.time_range

    t_start_us = t_min + 3_000_000
    t_end_us = t_start_us + VIDEO_DURATION * 1_000_000

    if t_end_us > t_max - 2_000_000:
        t_end_us = t_max - 2_000_000
        t_start_us = t_end_us - VIDEO_DURATION * 1_000_000

    print(f"Time: {t_start_us/1e6:.1f}s - {t_end_us/1e6:.1f}s")

    all_timestamps = np.arange(t_start_us, t_end_us, 100_000, dtype=np.int64)
    print(f"Loading {len(all_timestamps)} frames...")

    camera_features = [
        avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
    ]

    all_cam_frames = []
    for cam_feature in camera_features:
        camera = avdi.get_clip_feature(clip_id, cam_feature, maybe_stream=True)
        frames, _ = camera.decode_images_from_timestamps(all_timestamps)
        all_cam_frames.append(frames)

    all_cam_frames = np.stack(all_cam_frames, axis=1)
    print(f"Frames: {all_cam_frames.shape}")

    inference_times = np.arange(
        t_start_us + INFERENCE_INTERVAL * 1_000_000,
        t_end_us - 1_000_000,
        INFERENCE_INTERVAL * 1_000_000
    ).astype(np.int64)

    print(f"Running {len(inference_times)} inferences...")

    inference_results = []
    for i, t0_us in enumerate(inference_times):
        print(f"  [{i+1}/{len(inference_times)}] t={t0_us/1e6:.1f}s")

        # Clear cache before each inference
        torch.cuda.empty_cache()
        gc.collect()

        data = load_physical_aiavdataset(
            clip_id=clip_id,
            t0_us=int(t0_us),
            avdi=avdi,
            num_frames=4
        )

        messages = helper.create_message(data["image_frames"].flatten(0, 1))
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }
        model_inputs = helper.to_device(model_inputs, "cuda")

        torch.cuda.manual_seed_all(42 + i)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs, top_p=0.98, temperature=0.6,
                num_traj_samples=1, max_generation_length=256, return_extra=True,
            )

        pred_traj = pred_xyz.cpu().numpy()[0, 0, 0]
        gt_traj = data['ego_future_xyz'].cpu().numpy().squeeze()
        ade = np.mean(np.linalg.norm(pred_traj[:, :2] - gt_traj[:, :2], axis=1))
        coc_text = extra['cot'][0][0] if extra['cot'] else "Driving"

        inference_results.append({
            't0_us': int(t0_us),
            'coc': coc_text,
            'pred': pred_traj,
            'gt': gt_traj,
            'ade': float(ade)
        })
        print(f"    ADE: {ade:.2f}m | {coc_text[:50]}...")

        # Clear after inference
        del data, model_inputs, pred_xyz, pred_rot, extra
        torch.cuda.empty_cache()
        gc.collect()

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

    # Clear frames from memory
    del all_cam_frames
    gc.collect()

    return output_path


def main():
    import physical_ai_av
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper

    print("Initializing...")
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    clips = avdi.clip_index.index.tolist()
    print(f"Found {len(clips)} clips")

    print("Loading model with CPU offloading...")
    # Use device_map="auto" for automatic CPU/GPU distribution
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
        device_map="auto",  # Auto distribute to GPU + CPU
        max_memory={0: "20GiB", "cpu": "40GiB"},  # Reserve GPU memory, use CPU RAM
    )
    processor = helper.get_processor(model.tokenizer)
    print("Model loaded with offloading!")

    scenarios = [
        {'clip_id': clips[20], 'name': 'scenario_5s_d', 'title': 'Highway Cruise'},
        {'clip_id': clips[50], 'name': 'scenario_5s_e', 'title': 'Residential Area'},
        {'clip_id': clips[100], 'name': 'scenario_5s_f', 'title': 'Shopping District'},
    ]

    for s in scenarios:
        try:
            generate_scenario(s['clip_id'], s['name'], s['title'], avdi, model, processor)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()

    print("\nDone!")


if __name__ == "__main__":
    main()
