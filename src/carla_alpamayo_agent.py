"""
CARLA Alpamayo Agent
Main agent class that integrates Alpamayo VLA model with CARLA simulator
"""

import carla
import numpy as np
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .alpamayo_wrapper import AlpamayoWrapper, AlpamayoOutput
from .sensor_manager import SensorManager


@dataclass
class AgentConfig:
    """Configuration for CARLA Alpamayo Agent"""
    # CARLA connection
    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0

    # Vehicle settings
    vehicle_filter: str = "vehicle.tesla.model3"
    spawn_point_index: int = 0

    # Alpamayo settings
    model_name: str = "nvidia/Alpamayo-R1-10B"
    use_dummy_model: bool = False  # Use dummy for testing without GPU

    # Control settings
    target_fps: float = 10.0  # Target inference FPS
    max_speed: float = 30.0   # Max speed in km/h


class CarlaAlpamayoAgent:
    """
    Autonomous driving agent using NVIDIA Alpamayo VLA model in CARLA

    Features:
    - Chain-of-thought reasoning for driving decisions
    - Multi-sensor fusion support
    - Configurable control modes
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize CARLA Alpamayo Agent

        Args:
            config: Agent configuration
        """
        self.config = config or AgentConfig()

        # CARLA components
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.vehicle: Optional[carla.Actor] = None
        self.spectator: Optional[carla.Actor] = None

        # Components
        self.sensor_manager: Optional[SensorManager] = None
        self.alpamayo: Optional[AlpamayoWrapper] = None

        # State
        self.is_running = False
        self.frame_count = 0
        self.last_output: Optional[AlpamayoOutput] = None

    def connect(self) -> None:
        """Connect to CARLA server"""
        print(f"Connecting to CARLA at {self.config.host}:{self.config.port}...")

        self.client = carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(self.config.timeout)

        self.world = self.client.get_world()
        self.spectator = self.world.get_spectator()

        print(f"Connected to CARLA {self.client.get_server_version()}")
        print(f"Map: {self.world.get_map().name}")

    def spawn_vehicle(self) -> carla.Actor:
        """Spawn ego vehicle"""
        blueprint_library = self.world.get_blueprint_library()

        # Get vehicle blueprint
        blueprints = blueprint_library.filter(self.config.vehicle_filter)
        if not blueprints:
            blueprints = blueprint_library.filter("vehicle.*")
        blueprint = blueprints[0]

        # Get spawn point
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available")

        spawn_index = min(self.config.spawn_point_index, len(spawn_points) - 1)
        spawn_point = spawn_points[spawn_index]

        # Spawn vehicle
        self.vehicle = self.world.spawn_actor(blueprint, spawn_point)
        print(f"Spawned vehicle: {blueprint.id} at {spawn_point.location}")

        return self.vehicle

    def setup_sensors(self) -> None:
        """Setup vehicle sensors"""
        self.sensor_manager = SensorManager(self.world, self.vehicle)
        self.sensor_manager.spawn_default_sensors()

    def load_alpamayo(self) -> None:
        """Load Alpamayo model"""
        self.alpamayo = AlpamayoWrapper(model_name=self.config.model_name)

        if not self.config.use_dummy_model:
            print("Loading Alpamayo model (this may take several minutes)...")
            self.alpamayo.load_model()
        else:
            print("Using dummy model for testing (no GPU required)")

    def initialize(self) -> None:
        """Full initialization sequence"""
        self.connect()
        self.spawn_vehicle()
        self.setup_sensors()
        self.load_alpamayo()

        # Set synchronous mode for deterministic simulation
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / 20.0  # 20 FPS simulation
        self.world.apply_settings(settings)

        print("Agent initialized successfully!")

    def get_vehicle_state(self) -> Dict:
        """Get current vehicle state"""
        velocity = self.vehicle.get_velocity()
        transform = self.vehicle.get_transform()

        speed_ms = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_kmh = speed_ms * 3.6

        return {
            "speed_kmh": speed_kmh,
            "speed_ms": speed_ms,
            "location": {
                "x": transform.location.x,
                "y": transform.location.y,
                "z": transform.location.z,
            },
            "rotation": {
                "pitch": transform.rotation.pitch,
                "yaw": transform.rotation.yaw,
                "roll": transform.rotation.roll,
            },
        }

    def step(self, navigation_command: str = "follow the road") -> AlpamayoOutput:
        """
        Execute one control step

        Args:
            navigation_command: High-level navigation instruction

        Returns:
            Alpamayo model output with control commands
        """
        # Get sensor data
        image = self.sensor_manager.get_camera_image()
        if image is None:
            # Return safe stop if no image
            return AlpamayoOutput(
                steering=0.0,
                throttle=0.0,
                brake=1.0,
                reasoning="No camera data available",
            )

        # Get vehicle state
        state = self.get_vehicle_state()

        # Run Alpamayo inference
        if self.config.use_dummy_model:
            output = self.alpamayo.predict_dummy(
                image=image,
                navigation_command=navigation_command,
                speed_limit=self.config.max_speed,
                current_speed=state["speed_kmh"],
            )
        else:
            output = self.alpamayo.predict(
                image=image,
                navigation_command=navigation_command,
                speed_limit=self.config.max_speed,
                current_speed=state["speed_kmh"],
            )

        # Apply control
        control = carla.VehicleControl(
            throttle=output.throttle,
            steer=output.steering,
            brake=output.brake,
        )
        self.vehicle.apply_control(control)

        self.last_output = output
        self.frame_count += 1

        return output

    def update_spectator(self) -> None:
        """Move spectator camera to follow vehicle"""
        if self.vehicle and self.spectator:
            transform = self.vehicle.get_transform()
            spectator_transform = carla.Transform(
                carla.Location(
                    x=transform.location.x - 8 * np.cos(np.radians(transform.rotation.yaw)),
                    y=transform.location.y - 8 * np.sin(np.radians(transform.rotation.yaw)),
                    z=transform.location.z + 4,
                ),
                carla.Rotation(
                    pitch=-15,
                    yaw=transform.rotation.yaw,
                )
            )
            self.spectator.set_transform(spectator_transform)

    def run(
        self,
        max_frames: int = 1000,
        navigation_command: str = "follow the road",
        verbose: bool = True,
    ) -> None:
        """
        Run autonomous driving loop

        Args:
            max_frames: Maximum frames to run
            navigation_command: High-level navigation instruction
            verbose: Print status updates
        """
        print(f"Starting autonomous driving for {max_frames} frames...")
        print(f"Navigation: {navigation_command}")
        print("Press Ctrl+C to stop")

        self.is_running = True
        frame_time = 1.0 / self.config.target_fps

        try:
            for frame in range(max_frames):
                if not self.is_running:
                    break

                start_time = time.time()

                # Tick simulation
                self.world.tick()

                # Execute control step
                output = self.step(navigation_command)

                # Update spectator
                self.update_spectator()

                # Print status
                if verbose and frame % 10 == 0:
                    state = self.get_vehicle_state()
                    print(f"Frame {frame}: Speed={state['speed_kmh']:.1f} km/h, "
                          f"Steer={output.steering:.2f}, Throttle={output.throttle:.2f}")

                # Maintain target FPS
                elapsed = time.time() - start_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.is_running = False

    def cleanup(self) -> None:
        """Cleanup all resources"""
        print("Cleaning up...")

        # Destroy sensors
        if self.sensor_manager:
            self.sensor_manager.destroy_all()

        # Destroy vehicle
        if self.vehicle and self.vehicle.is_alive:
            self.vehicle.destroy()
            print("Destroyed vehicle")

        # Reset world settings
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

        print("Cleanup complete")

    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
        return False
