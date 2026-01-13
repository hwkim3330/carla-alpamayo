#!/usr/bin/env python3
"""
Example: Run CARLA Alpamayo Agent with Recording
Records camera images and control outputs for analysis
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, '..')

import numpy as np
from PIL import Image

from src.carla_alpamayo_agent import CarlaAlpamayoAgent, AgentConfig


def main():
    parser = argparse.ArgumentParser(description="Run CARLA Alpamayo with Recording")
    parser.add_argument("--host", default="localhost", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--frames", type=int, default=500, help="Max frames")
    parser.add_argument("--dummy", action="store_true", help="Use dummy model")
    parser.add_argument("--output", default="recordings", help="Output directory")
    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    print(f"Recording to: {output_dir}")

    # Configure agent
    config = AgentConfig(
        host=args.host,
        port=args.port,
        use_dummy_model=args.dummy,
    )

    # Data storage
    recording_data = {
        "config": {
            "host": args.host,
            "port": args.port,
            "frames": args.frames,
            "timestamp": timestamp,
        },
        "frames": [],
    }

    # Run agent with recording
    agent = CarlaAlpamayoAgent(config)
    try:
        agent.initialize()

        for frame_idx in range(args.frames):
            # Tick simulation
            agent.world.tick()

            # Get sensor data
            image = agent.sensor_manager.get_camera_image()
            if image is None:
                continue

            # Run inference
            output = agent.step("follow the road")

            # Save image
            img_path = images_dir / f"frame_{frame_idx:06d}.jpg"
            Image.fromarray(image).save(img_path, quality=90)

            # Record frame data
            state = agent.get_vehicle_state()
            frame_data = {
                "frame": frame_idx,
                "timestamp": time.time(),
                "state": state,
                "control": {
                    "steering": output.steering,
                    "throttle": output.throttle,
                    "brake": output.brake,
                },
                "reasoning": output.reasoning[:500] if output.reasoning else "",
                "image_file": f"images/frame_{frame_idx:06d}.jpg",
            }
            recording_data["frames"].append(frame_data)

            # Update spectator
            agent.update_spectator()

            # Progress
            if frame_idx % 50 == 0:
                print(f"Frame {frame_idx}/{args.frames} - "
                      f"Speed: {state['speed_kmh']:.1f} km/h")

    except KeyboardInterrupt:
        print("\nStopping recording...")
    finally:
        # Save recording data
        json_path = output_dir / "recording.json"
        with open(json_path, "w") as f:
            json.dump(recording_data, f, indent=2)
        print(f"Saved recording data to: {json_path}")

        agent.cleanup()

    print(f"Recording complete: {len(recording_data['frames'])} frames")


if __name__ == "__main__":
    main()
