"""
CARLA Sensor Manager
Handles sensor spawning, data collection, and synchronization
"""

import carla
import numpy as np
import queue
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class SensorConfig:
    """Configuration for a sensor"""
    sensor_type: str
    transform: carla.Transform
    attributes: Dict[str, str] = field(default_factory=dict)


class SensorManager:
    """
    Manages CARLA sensors for autonomous driving
    Handles camera, lidar, and other sensor data collection
    """

    # Default sensor configurations
    DEFAULT_CONFIGS = {
        "front_camera": SensorConfig(
            sensor_type="sensor.camera.rgb",
            transform=carla.Transform(
                carla.Location(x=2.0, z=1.4),
                carla.Rotation(pitch=-10)
            ),
            attributes={
                "image_size_x": "1280",
                "image_size_y": "720",
                "fov": "90",
            }
        ),
        "front_camera_wide": SensorConfig(
            sensor_type="sensor.camera.rgb",
            transform=carla.Transform(
                carla.Location(x=2.0, z=1.4),
                carla.Rotation(pitch=-10)
            ),
            attributes={
                "image_size_x": "1920",
                "image_size_y": "1080",
                "fov": "110",
            }
        ),
        "semantic_camera": SensorConfig(
            sensor_type="sensor.camera.semantic_segmentation",
            transform=carla.Transform(
                carla.Location(x=2.0, z=1.4),
                carla.Rotation(pitch=-10)
            ),
            attributes={
                "image_size_x": "1280",
                "image_size_y": "720",
                "fov": "90",
            }
        ),
        "depth_camera": SensorConfig(
            sensor_type="sensor.camera.depth",
            transform=carla.Transform(
                carla.Location(x=2.0, z=1.4),
                carla.Rotation(pitch=-10)
            ),
            attributes={
                "image_size_x": "1280",
                "image_size_y": "720",
                "fov": "90",
            }
        ),
        "lidar": SensorConfig(
            sensor_type="sensor.lidar.ray_cast",
            transform=carla.Transform(
                carla.Location(x=0.0, z=2.4)
            ),
            attributes={
                "channels": "64",
                "range": "100",
                "points_per_second": "1200000",
                "rotation_frequency": "20",
            }
        ),
        "gnss": SensorConfig(
            sensor_type="sensor.other.gnss",
            transform=carla.Transform(
                carla.Location(x=0.0, z=0.0)
            ),
            attributes={}
        ),
        "imu": SensorConfig(
            sensor_type="sensor.other.imu",
            transform=carla.Transform(
                carla.Location(x=0.0, z=0.0)
            ),
            attributes={}
        ),
    }

    def __init__(self, world: carla.World, vehicle: carla.Actor):
        """
        Initialize sensor manager

        Args:
            world: CARLA world instance
            vehicle: Vehicle to attach sensors to
        """
        self.world = world
        self.vehicle = vehicle
        self.blueprint_library = world.get_blueprint_library()

        self.sensors: Dict[str, carla.Actor] = {}
        self.data_queues: Dict[str, queue.Queue] = {}
        self.latest_data: Dict[str, any] = {}

    def spawn_sensor(
        self,
        name: str,
        config: Optional[SensorConfig] = None,
    ) -> carla.Actor:
        """
        Spawn a sensor and attach to vehicle

        Args:
            name: Sensor identifier (use DEFAULT_CONFIGS key or custom)
            config: Custom sensor configuration (optional)

        Returns:
            Spawned sensor actor
        """
        if config is None:
            if name not in self.DEFAULT_CONFIGS:
                raise ValueError(f"Unknown sensor: {name}. Provide custom config.")
            config = self.DEFAULT_CONFIGS[name]

        # Get blueprint
        blueprint = self.blueprint_library.find(config.sensor_type)

        # Set attributes
        for attr, value in config.attributes.items():
            if blueprint.has_attribute(attr):
                blueprint.set_attribute(attr, value)

        # Spawn sensor
        sensor = self.world.spawn_actor(
            blueprint,
            config.transform,
            attach_to=self.vehicle,
        )

        # Create data queue
        self.data_queues[name] = queue.Queue()

        # Set up listener
        sensor.listen(lambda data, n=name: self._on_sensor_data(n, data))

        self.sensors[name] = sensor
        print(f"Spawned sensor: {name} ({config.sensor_type})")

        return sensor

    def spawn_default_sensors(self) -> Dict[str, carla.Actor]:
        """Spawn a default set of sensors for Alpamayo"""
        sensors_to_spawn = ["front_camera", "gnss", "imu"]

        for name in sensors_to_spawn:
            self.spawn_sensor(name)

        return self.sensors

    def _on_sensor_data(self, name: str, data: any) -> None:
        """Callback for sensor data"""
        self.latest_data[name] = data

        try:
            self.data_queues[name].put_nowait(data)
        except queue.Full:
            # Drop old data if queue is full
            try:
                self.data_queues[name].get_nowait()
                self.data_queues[name].put_nowait(data)
            except queue.Empty:
                pass

    def get_camera_image(
        self,
        name: str = "front_camera",
        timeout: float = 1.0,
    ) -> Optional[np.ndarray]:
        """
        Get latest camera image as numpy array

        Args:
            name: Camera sensor name
            timeout: Timeout in seconds

        Returns:
            RGB image as numpy array (H, W, 3)
        """
        try:
            data = self.data_queues[name].get(timeout=timeout)
            return self._process_camera_image(data)
        except queue.Empty:
            # Return latest data if available
            if name in self.latest_data:
                return self._process_camera_image(self.latest_data[name])
            return None

    def _process_camera_image(self, image: carla.Image) -> np.ndarray:
        """Convert CARLA image to numpy array"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        # Convert BGRA to RGB
        return array[:, :, :3][:, :, ::-1].copy()

    def get_gnss_data(self, name: str = "gnss") -> Optional[Dict]:
        """Get latest GNSS data"""
        if name in self.latest_data:
            data = self.latest_data[name]
            return {
                "latitude": data.latitude,
                "longitude": data.longitude,
                "altitude": data.altitude,
            }
        return None

    def get_imu_data(self, name: str = "imu") -> Optional[Dict]:
        """Get latest IMU data"""
        if name in self.latest_data:
            data = self.latest_data[name]
            return {
                "accelerometer": {
                    "x": data.accelerometer.x,
                    "y": data.accelerometer.y,
                    "z": data.accelerometer.z,
                },
                "gyroscope": {
                    "x": data.gyroscope.x,
                    "y": data.gyroscope.y,
                    "z": data.gyroscope.z,
                },
                "compass": data.compass,
            }
        return None

    def destroy_all(self) -> None:
        """Destroy all sensors"""
        for name, sensor in self.sensors.items():
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
                print(f"Destroyed sensor: {name}")

        self.sensors.clear()
        self.data_queues.clear()
        self.latest_data.clear()
