#!/usr/bin/env python3
"""
Example: Run CARLA Alpamayo Agent
Basic example of running autonomous driving with Alpamayo VLA model
"""

import argparse
import sys
sys.path.insert(0, '..')

from src.carla_alpamayo_agent import CarlaAlpamayoAgent, AgentConfig


def main():
    parser = argparse.ArgumentParser(description="Run CARLA Alpamayo Agent")
    parser.add_argument("--host", default="localhost", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--frames", type=int, default=1000, help="Max frames to run")
    parser.add_argument("--dummy", action="store_true", help="Use dummy model (no GPU)")
    parser.add_argument("--command", default="follow the road",
                        help="Navigation command")
    parser.add_argument("--vehicle", default="vehicle.tesla.model3",
                        help="Vehicle filter")
    args = parser.parse_args()

    # Configure agent
    config = AgentConfig(
        host=args.host,
        port=args.port,
        vehicle_filter=args.vehicle,
        use_dummy_model=args.dummy,
    )

    # Run agent
    with CarlaAlpamayoAgent(config) as agent:
        agent.run(
            max_frames=args.frames,
            navigation_command=args.command,
            verbose=True,
        )


if __name__ == "__main__":
    main()
