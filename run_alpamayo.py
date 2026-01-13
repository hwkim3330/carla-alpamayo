#!/usr/bin/env python3
"""
Main script to run Alpamayo on CARLA 0.9.16
Implements Pure Pursuit + PID control with 4-camera setup
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
import signal

def check_carla_running(host='localhost', port=2000):
    """Check if CARLA server is running"""
    import socket
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    
    return result == 0

def start_carla_server(carla_path=None):
    """Start CARLA server if not running"""
    if check_carla_running():
        print("✅ CARLA server already running")
        return None
    
    print("🚀 Starting CARLA server...")
    
    if carla_path is None:
        # Try to find CARLA in common locations
        possible_paths = [
            "/opt/carla",
            "~/CARLA",
            "./CARLA",
            os.environ.get("CARLA_ROOT", "")
        ]
        
        for path in possible_paths:
            path = os.path.expanduser(path)
            if os.path.exists(path):
                carla_path = path
                break
    
    if carla_path is None:
        print("❌ CARLA not found. Please set CARLA_ROOT environment variable")
        return None
    
    # Start CARLA
    carla_executable = os.path.join(carla_path, "CarlaUE4.sh")
    if not os.path.exists(carla_executable):
        carla_executable = os.path.join(carla_path, "CarlaUE4.exe")
    
    process = subprocess.Popen([
        carla_executable,
        "-quality-level=Low",
        "-RenderOffScreen"
    ])
    
    # Wait for server to start
    print("⏳ Waiting for CARLA to start...")
    for _ in range(30):
        if check_carla_running():
            print("✅ CARLA server started")
            return process
        time.sleep(1)
    
    print("❌ Failed to start CARLA server")
    return None

def main():
    parser = argparse.ArgumentParser(description='Run Alpamayo VLA on CARLA 0.9.16')
    parser.add_argument('--host', default='localhost', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--town', default='Town05', choices=[
        'Town01', 'Town02', 'Town03', 'Town04', 'Town05',
        'Town06', 'Town07', 'Town10', 'Town10HD', 'Town11', 'Town12'
    ], help='CARLA map/town (Town05 recommended for testing)')
    parser.add_argument('--weather', default='ClearNoon', choices=[
        'ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon',
        'SoftRainNoon', 'MidRainyNoon', 'HardRainNoon',
        'ClearSunset', 'CloudySunset', 'WetSunset'
    ], help='Weather preset')
    parser.add_argument('--duration', type=float, default=300, 
                       help='Duration in seconds')
    parser.add_argument('--command', default='Drive safely and follow traffic rules',
                       help='Natural language command (Korean/English)')
    parser.add_argument('--spawn-point', type=int, default=None,
                       help='Spawn point index')
    parser.add_argument('--auto-start-carla', action='store_true',
                       help='Automatically start CARLA if not running')
    parser.add_argument('--traffic', type=int, default=10,
                       help='Number of NPC vehicles to spawn')
    parser.add_argument('--test-scenario', choices=['straight', 'intersection', 'curve'],
                       help='Predefined test scenario')
    
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════╗
    ║   🚗 ALPAMAYO on CARLA 0.9.16 Simulator 🚗  ║
    ║   NVIDIA's VLA Model for Autonomous         ║
    ║   Driving with Chain-of-Causation           ║
    ║   Pure Pursuit + PID Control @ 10Hz         ║
    ╚══════════════════════════════════════════════╝
    """)
    
    # Check/start CARLA
    carla_process = None
    if args.auto_start_carla:
        carla_process = start_carla_server()
    elif not check_carla_running(args.host, args.port):
        print(f"❌ CARLA server not running at {args.host}:{args.port}")
        print("Start CARLA manually or use --auto-start-carla flag")
        return
    
    try:
        # Import here to avoid errors if CARLA not installed
        import carla
        from carla_integration import AlpamayoCarlaAgent
        
        # Connect to CARLA
        print(f"\n📡 Connecting to CARLA at {args.host}:{args.port}")
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        
        # Load town
        print(f"🗺️ Loading {args.town}...")
        world = client.load_world(args.town)
        
        # Set weather
        print(f"🌤️ Setting weather to {args.weather}")
        weather_presets = {
            'ClearNoon': carla.WeatherParameters.ClearNoon,
            'CloudyNoon': carla.WeatherParameters.CloudyNoon,
            'WetNoon': carla.WeatherParameters.WetNoon,
            'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
            'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
            'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
            'HardRainNoon': carla.WeatherParameters.HardRainNoon,
            'ClearSunset': carla.WeatherParameters.ClearSunset,
            'CloudySunset': carla.WeatherParameters.CloudySunset,
            'WetSunset': carla.WeatherParameters.WetSunset,
        }
        world.set_weather(weather_presets[args.weather])
        
        # Spawn traffic if requested
        if args.traffic > 0:
            spawn_traffic(world, args.traffic)
            print(f"🚦 Spawned {args.traffic} NPC vehicles")
        
        # Create Alpamayo agent (handles sync mode internally)
        print("\n🤖 Initializing Alpamayo Agent with 4-camera rig...")
        agent = AlpamayoCarlaAgent(args.host, args.port)
        
        # Spawn at specific point if requested
        if args.spawn_point is not None:
            spawn_points = world.get_map().get_spawn_points()
            if 0 <= args.spawn_point < len(spawn_points):
                agent.spawn_vehicle(spawn_points[args.spawn_point])
        
        # Setup test scenario if specified
        if args.test_scenario:
            setup_test_scenario(world, agent, args.test_scenario)
            print(f"🎬 Test scenario: {args.test_scenario}")
        
        # Run autonomous driving
        print(f"\n🚗 Starting Alpamayo autonomous driving...")
        print(f"📝 Command: {args.command}")
        print(f"⏱️ Duration: {args.duration} seconds")
        print(f"🎮 Control: Pure Pursuit (steering) + PID (speed)")
        print(f"📊 Target: 10Hz control loop")
        print("-" * 50)
        
        # Handle graceful shutdown
        def signal_handler(sig, frame):
            print("\n🛑 Stopping Alpamayo...")
            agent.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        agent.run(duration=args.duration, command=args.command)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure required packages are installed:")
        print("  pip install carla==0.9.16")
        print("  pip install torch transformers opencv-python")
    except ConnectionError:
        print(f"❌ Cannot connect to CARLA at {args.host}:{args.port}")
        print("Make sure CARLA 0.9.16 is running or use --auto-start-carla")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        try:
            if 'agent' in locals():
                agent.cleanup()
        except:
            pass
            
        if carla_process:
            print("\n🛑 Stopping CARLA server...")
            carla_process.terminate()
            carla_process.wait()

def spawn_traffic(world, num_vehicles):
    """Spawn NPC traffic vehicles"""
    try:
        blueprints = world.get_blueprint_library().filter('vehicle.*')
        spawn_points = world.get_map().get_spawn_points()
        
        import random
        random.shuffle(spawn_points)
        
        for i in range(min(num_vehicles, len(spawn_points))):
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            
            vehicle = world.try_spawn_actor(blueprint, spawn_points[i])
            if vehicle:
                vehicle.set_autopilot(True)
    except Exception as e:
        print(f"Warning: Could not spawn all traffic: {e}")

def setup_test_scenario(world, agent, scenario):
    """Setup predefined test scenarios"""
    if scenario == 'straight':
        # Straight road test
        pass  # Use default spawn
    elif scenario == 'intersection':
        # Find intersection spawn point
        spawn_points = world.get_map().get_spawn_points()
        # Look for spawn near intersection (implementation depends on map)
    elif scenario == 'curve':
        # Find curved road spawn point
        pass

if __name__ == "__main__":
    main()