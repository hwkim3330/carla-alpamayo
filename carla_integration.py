#!/usr/bin/env python3
"""
CARLA 0.9.16 Integration for Alpamayo VLA Model  
Multi-camera sensor setup with Pure Pursuit + PID control
Based on NVIDIA Alpamayo-R1-10B specifications
"""

import carla
import numpy as np
import cv2
import time
import math
from typing import Dict, List, Optional, Tuple
from collections import deque
import threading
import queue
from dataclasses import dataclass

@dataclass
class PIDController:
    """PID controller for speed control"""
    kp: float = 1.0
    ki: float = 0.1
    kd: float = 0.01
    target: float = 0.0
    error_sum: float = 0.0
    last_error: float = 0.0
    
    def update(self, current, dt=0.1):
        error = self.target - current
        self.error_sum += error * dt
        
        # Anti-windup
        self.error_sum = np.clip(self.error_sum, -10, 10)
        
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        self.last_error = error
        
        output = self.kp * error + self.ki * self.error_sum + self.kd * derivative
        return np.clip(output, -1.0, 1.0)

class PurePursuitController:
    """Pure Pursuit controller for lateral control"""
    
    def __init__(self, k=1.0, Lfc=5.0, L=2.875):  # L is wheelbase for Tesla Model 3
        self.k = k  # Look-ahead gain
        self.Lfc = Lfc  # Look-ahead distance constant
        self.L = L  # Vehicle wheelbase
        
    def calculate_steering(self, waypoints, vehicle_transform, current_speed):
        """
        Calculate steering angle using Pure Pursuit algorithm
        
        Args:
            waypoints: Future trajectory points in world coordinates
            vehicle_transform: Current vehicle transform
            current_speed: Current vehicle speed in m/s
            
        Returns:
            Steering angle in [-1, 1] range
        """
        if len(waypoints) < 2:
            return 0.0
            
        # Dynamic look-ahead distance based on speed
        Ld = self.k * abs(current_speed) + self.Lfc
        Ld = max(Ld, 2.0)  # Minimum look-ahead
        
        # Find target waypoint
        target_idx = self._find_target_waypoint(waypoints, vehicle_transform, Ld)
        
        if target_idx < 0:
            return 0.0
            
        # Convert target to vehicle coordinates
        target_world = waypoints[target_idx]
        target_local = self._world_to_vehicle(target_world, vehicle_transform)
        
        # Pure pursuit formula
        alpha = math.atan2(target_local[1], target_local[0])
        steering = math.atan(2.0 * self.L * math.sin(alpha) / Ld)
        
        # Normalize to [-1, 1]
        return np.clip(steering / 0.7, -1.0, 1.0)  # 0.7 rad max steering
        
    def _find_target_waypoint(self, waypoints, vehicle_transform, look_ahead_dist):
        """Find the waypoint at look-ahead distance"""
        vehicle_location = vehicle_transform.location
        
        for i, waypoint in enumerate(waypoints):
            dist = math.sqrt(
                (waypoint[0] - vehicle_location.x)**2 +
                (waypoint[1] - vehicle_location.y)**2
            )
            if dist >= look_ahead_dist:
                return i
                
        return len(waypoints) - 1  # Return last waypoint if none found
        
    def _world_to_vehicle(self, world_point, vehicle_transform):
        """Transform world coordinates to vehicle-relative coordinates"""
        # Get vehicle position and orientation
        x = vehicle_transform.location.x
        y = vehicle_transform.location.y
        yaw = math.radians(vehicle_transform.rotation.yaw)
        
        # Translation
        dx = world_point[0] - x
        dy = world_point[1] - y
        
        # Rotation to vehicle frame
        x_vehicle = dx * math.cos(-yaw) - dy * math.sin(-yaw)
        y_vehicle = dx * math.sin(-yaw) + dy * math.cos(-yaw)
        
        return [x_vehicle, y_vehicle]

class CARLASensorManager:
    """Manage multi-camera sensors in CARLA"""
    
    def __init__(self, vehicle, world):
        self.vehicle = vehicle
        self.world = world
        self.sensors = {}
        self.sensor_data = {}
        self.data_lock = threading.Lock()
        
        # Camera configurations matching Alpamayo spec for CARLA 0.9.16
        # 4 cameras: front-wide, front-tele, cross-left, cross-right
        self.camera_configs = {
            'front_wide': {
                'transform': carla.Transform(
                    carla.Location(x=1.5, z=1.6),  # Forward facing, hood mount
                    carla.Rotation(pitch=-5)  # Slight downward tilt
                ),
                'fov': 120,  # Wide angle
                'image_size': (1920, 1080)
            },
            'front_tele': {
                'transform': carla.Transform(
                    carla.Location(x=1.5, z=1.6),  # Same position as wide
                    carla.Rotation(pitch=-5)
                ),
                'fov': 30,  # Telephoto for distant objects
                'image_size': (1920, 1080)
            },
            'cross_left': {
                'transform': carla.Transform(
                    carla.Location(x=1.0, y=-0.5, z=1.6),  # Left side
                    carla.Rotation(yaw=-45, pitch=-10)  # 45° left for intersection
                ),
                'fov': 90,
                'image_size': (1920, 1080)
            },
            'cross_right': {
                'transform': carla.Transform(
                    carla.Location(x=1.0, y=0.5, z=1.6),  # Right side
                    carla.Rotation(yaw=45, pitch=-10)  # 45° right for intersection
                ),
                'fov': 90,
                'image_size': (1920, 1080)
            }
        }
        
        # History buffers (4 frames per camera at 10Hz)
        self.history_length = 4
        self.image_history = {name: deque(maxlen=self.history_length) 
                             for name in self.camera_configs}
        
        self.setup_sensors()
    
    def setup_sensors(self):
        """Setup all camera sensors"""
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        
        for name, config in self.camera_configs.items():
            # Configure camera
            camera_bp.set_attribute('image_size_x', str(config['image_size'][0]))
            camera_bp.set_attribute('image_size_y', str(config['image_size'][1]))
            camera_bp.set_attribute('fov', str(config['fov']))
            
            # Spawn camera
            camera = self.world.spawn_actor(
                camera_bp,
                config['transform'],
                attach_to=self.vehicle
            )
            
            # Setup callback
            camera.listen(lambda image, n=name: self._on_camera_data(image, n))
            self.sensors[name] = camera
            
        # Also setup IMU for egomotion
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu = self.world.spawn_actor(
            imu_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        imu.listen(self._on_imu_data)
        self.sensors['imu'] = imu
        
        print(f"✅ Setup {len(self.camera_configs)} cameras + IMU")
    
    def _on_camera_data(self, image, camera_name):
        """Process camera data and maintain 10Hz buffer"""
        # Convert to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        
        # Resize to model input size (320x576) for efficiency
        array_resized = cv2.resize(array, (576, 320))  # Width x Height
        
        # Store in history with timestamp
        with self.data_lock:
            self.image_history[camera_name].append({
                'data': array_resized,
                'timestamp': image.timestamp,
                'frame': image.frame
            })
    
    def _on_imu_data(self, imu_data):
        """Process IMU data"""
        with self.data_lock:
            self.sensor_data['imu'] = {
                'accelerometer': [
                    imu_data.accelerometer.x,
                    imu_data.accelerometer.y,
                    imu_data.accelerometer.z
                ],
                'gyroscope': [
                    imu_data.gyroscope.x,
                    imu_data.gyroscope.y,
                    imu_data.gyroscope.z
                ],
                'timestamp': imu_data.timestamp
            }
    
    def get_sensor_data(self) -> Dict:
        """Get current sensor data for model input"""
        with self.data_lock:
            # Collect images from all cameras
            images = []
            for camera_name in ['front_wide', 'front_tele', 'cross_left', 'cross_right']:
                camera_history = self.image_history[camera_name]
                if len(camera_history) > 0:
                    # Get latest frames
                    for frame in camera_history:
                        images.append(frame['data'])
            
            # Get vehicle state
            transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            
            # Create egomotion history (simplified)
            ego_history = self._compute_ego_history(transform, velocity)
            
            return {
                'images': images,
                'ego_history': ego_history,
                'transform': transform,
                'velocity': velocity
            }
    
    def _compute_ego_history(self, transform, velocity):
        """Compute egomotion history matrix for Alpamayo input"""
        # Maintain history buffer if not exists
        if not hasattr(self, 'ego_history_buffer'):
            self.ego_history_buffer = deque(maxlen=16)
        
        # Current position
        x, y, z = transform.location.x, transform.location.y, transform.location.z
        
        # Rotation matrix from Euler angles (CARLA uses degrees)
        roll, pitch, yaw = np.radians([
            transform.rotation.roll,
            transform.rotation.pitch,
            transform.rotation.yaw
        ])
        
        # Convert to rotation matrix
        R = self._euler_to_rotation_matrix(roll, pitch, yaw)
        
        # Create ego state vector [x, y, z, R_flat]
        ego_state = np.zeros(12)
        ego_state[:3] = [x, y, z]
        ego_state[3:] = R.flatten()
        
        # Add to buffer with timestamp
        self.ego_history_buffer.append({
            'state': ego_state,
            'velocity': [velocity.x, velocity.y, velocity.z],
            'timestamp': time.time()
        })
        
        # Convert buffer to matrix format [1, 16, 12]
        ego_history = np.zeros((1, 16, 12))
        for i, entry in enumerate(self.ego_history_buffer):
            ego_history[0, i, :] = entry['state']
        
        return ego_history
    
    def _euler_to_rotation_matrix(self, roll, pitch, yaw):
        """Convert Euler angles to rotation matrix"""
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        return R_z @ R_y @ R_x
    
    def destroy(self):
        """Clean up sensors"""
        for sensor in self.sensors.values():
            sensor.destroy()

class AlpamayoCarlaAgent:
    """Main agent that runs Alpamayo model in CARLA 0.9.16"""
    
    def __init__(self, host='localhost', port=2000):
        print("🚗 Initializing Alpamayo CARLA Agent for 0.9.16...")
        
        # Connect to CARLA
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Set synchronous mode for stable 10Hz operation
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1  # 10Hz
        self.world.apply_settings(settings)
        
        # Spawn vehicle
        self.vehicle = None
        self.sensor_manager = None
        self.spawn_vehicle()
        
        # Initialize Alpamayo model
        from alpamayo_model import create_alpamayo_model
        self.model = create_alpamayo_model()
        
        # Initialize controllers
        self.pure_pursuit = PurePursuitController(k=1.0, Lfc=5.0)
        self.speed_pid = PIDController(kp=0.5, ki=0.1, kd=0.05)
        self.speed_pid.target = 30.0 / 3.6  # Convert km/h to m/s
        
        # Control parameters
        self.target_speed = 30.0  # km/h
        self.max_steering = 0.7
        
        # Trajectory buffer for smooth control
        self.current_trajectory = None
        self.trajectory_timestamp = 0
        
        # Reasoning trace buffer
        self.reasoning_history = deque(maxlen=10)
        
    def spawn_vehicle(self, spawn_point=None):
        """Spawn ego vehicle"""
        blueprint_library = self.world.get_blueprint_library()
        
        # Get Tesla Model 3 blueprint
        vehicle_bp = blueprint_library.filter('model3')[0]
        
        # Get spawn point
        if spawn_point is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = np.random.choice(spawn_points)
        
        # Spawn vehicle
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"✅ Spawned vehicle at {spawn_point.location}")
        
        # Setup sensors
        self.sensor_manager = CARLASensorManager(self.vehicle, self.world)
        
        # Let everything initialize
        time.sleep(2)
    
    def run_step(self, command: str = "Drive safely") -> Dict:
        """Execute one step of autonomous driving with Pure Pursuit + PID"""
        
        # Get sensor data
        sensor_data = self.sensor_manager.get_sensor_data()
        
        if len(sensor_data['images']) < 16:  # 4 cameras x 4 frames
            # Not enough sensor data yet, apply brake
            control = carla.VehicleControl(throttle=0, brake=1.0, steer=0)
            self.vehicle.apply_control(control)
            return {'controls': {'throttle': 0, 'brake': 1, 'steering': 0}}
        
        # Get current vehicle state
        velocity = sensor_data['velocity']
        current_speed = math.sqrt(velocity.x**2 + velocity.y**2)  # m/s
        transform = sensor_data['transform']
        
        # Run Alpamayo model every 0.5 seconds (5 frames at 10Hz)
        current_time = time.time()
        if self.current_trajectory is None or \
           (current_time - self.trajectory_timestamp) > 0.5:
            
            # Run inference
            prediction = self.model.predict_trajectory(sensor_data, command)
            
            # Update trajectory
            self.current_trajectory = prediction['trajectory'].cpu().numpy()
            self.trajectory_timestamp = current_time
            
            # Store reasoning
            self.reasoning_history.append({
                'timestamp': current_time,
                'reasoning': prediction['reasoning'],
                'trajectory': self.current_trajectory
            })
        
        # Convert trajectory to world coordinates
        waypoints_world = self._trajectory_to_world(self.current_trajectory, transform)
        
        # Pure Pursuit for steering
        steer = self.pure_pursuit.calculate_steering(
            waypoints_world, 
            transform, 
            current_speed
        )
        
        # PID for speed control
        speed_error = self.speed_pid.target - current_speed
        if speed_error > 0:
            throttle = self.speed_pid.update(current_speed, dt=0.1)
            brake = 0.0
        else:
            throttle = 0.0
            brake = abs(self.speed_pid.update(current_speed, dt=0.1))
        
        # Safety checks
        if self._check_emergency_brake(sensor_data):
            throttle = 0.0
            brake = 1.0
            
        # Apply control
        control = carla.VehicleControl(
            throttle=float(np.clip(throttle, 0, 1)),
            brake=float(np.clip(brake, 0, 1)),
            steer=float(np.clip(steer, -1, 1))
        )
        self.vehicle.apply_control(control)
        
        return {
            'controls': {
                'throttle': control.throttle,
                'brake': control.brake,
                'steering': control.steer
            },
            'speed': current_speed * 3.6,  # km/h
            'trajectory': self.current_trajectory
        }
    
    def _trajectory_to_world(self, trajectory: np.ndarray, transform: carla.Transform):
        """Convert ego-relative trajectory to world coordinates"""
        waypoints_world = []
        
        # Get vehicle position and orientation
        x = transform.location.x
        y = transform.location.y
        z = transform.location.z
        yaw = math.radians(transform.rotation.yaw)
        
        # Transform each waypoint
        for waypoint in trajectory:
            if len(waypoint.shape) > 1:
                waypoint = waypoint[0]  # Handle batch dimension
                
            # Rotate and translate to world frame
            x_world = x + waypoint[0] * math.cos(yaw) - waypoint[1] * math.sin(yaw)
            y_world = y + waypoint[0] * math.sin(yaw) + waypoint[1] * math.cos(yaw)
            z_world = z + waypoint[2] if len(waypoint) > 2 else z
            
            waypoints_world.append([x_world, y_world, z_world])
            
        return waypoints_world
    
    def _check_emergency_brake(self, sensor_data):
        """Check if emergency braking is needed"""
        # Simple collision detection (would use radar/lidar in real implementation)
        # For now, return False
        return False
    
    def visualize_trajectory(self, trajectory: np.ndarray):
        """Visualize predicted trajectory in CARLA with color gradient"""
        # Get current vehicle transform
        vehicle_transform = self.vehicle.get_transform()
        
        # Convert to world coordinates
        waypoints_world = self._trajectory_to_world(trajectory, vehicle_transform)
        
        # Draw waypoints with color gradient (green to yellow)
        for i, waypoint in enumerate(waypoints_world):
            # Color interpolation
            t = i / len(waypoints_world)
            r = int(255 * t)
            g = 255
            b = 0
            
            location = carla.Location(x=waypoint[0], y=waypoint[1], z=waypoint[2] + 0.5)
            
            # Draw sphere at waypoint
            self.world.debug.draw_point(
                location,
                size=0.1,
                color=carla.Color(r, g, b),
                life_time=0.5
            )
            
            # Draw line between waypoints
            if i > 0:
                prev_location = carla.Location(
                    x=waypoints_world[i-1][0],
                    y=waypoints_world[i-1][1],
                    z=waypoints_world[i-1][2] + 0.5
                )
                
                self.world.debug.draw_line(
                    prev_location,
                    location,
                    thickness=0.05,
                    color=carla.Color(r, g, b),
                    life_time=0.5
                )
    
    def run(self, duration: float = 60.0, command: str = "Drive safely and follow traffic rules"):
        """Main driving loop at 10Hz for CARLA 0.9.16"""
        print(f"\n🚀 Starting Alpamayo autonomous driving for {duration} seconds")
        print(f"📝 Command: {command}")
        print(f"🎮 Control: Pure Pursuit + PID")
        print(f"📸 Sensors: 4 cameras @ 10Hz")
        print("-" * 50)
        
        start_time = time.time()
        step = 0
        
        try:
            while time.time() - start_time < duration:
                # Tick world in sync mode
                self.world.tick()
                
                # Run one step
                result = self.run_step(command)
                
                # Visualize trajectory
                if 'trajectory' in result and result['trajectory'] is not None:
                    self.visualize_trajectory(result['trajectory'])
                
                # Print status every 10 steps (1 second)
                if step % 10 == 0:
                    print(f"\n[{time.time()-start_time:.1f}s] Step {step}:")
                    
                    if self.reasoning_history:
                        latest = self.reasoning_history[-1]
                        print(f"CoC: {latest['reasoning'][:100]}...")
                    
                    print(f"Speed: {result.get('speed', 0):.1f} km/h | "
                          f"Throttle: {result['controls']['throttle']:.2f} | "
                          f"Brake: {result['controls']['brake']:.2f} | "
                          f"Steering: {result['controls']['steering']:.2f}")
                
                step += 1
                
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        finally:
            # Restore async mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        if self.sensor_manager:
            self.sensor_manager.destroy()
        if self.vehicle:
            self.vehicle.destroy()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Alpamayo CARLA Agent')
    parser.add_argument('--host', default='localhost', help='CARLA host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA port')
    parser.add_argument('--duration', type=float, default=60, help='Duration in seconds')
    parser.add_argument('--command', default='Navigate to destination safely while following traffic rules',
                      help='Natural language command')
    
    args = parser.parse_args()
    
    # Create and run agent
    agent = AlpamayoCarlaAgent(host=args.host, port=args.port)
    agent.run(duration=args.duration, command=args.command)

if __name__ == "__main__":
    main()