"""
CARLA-Alpamayo Integration Package
NVIDIA Alpamayo VLA model integration with CARLA simulator
"""

__version__ = "0.1.0"
__author__ = "hwkim3330"

from .carla_alpamayo_agent import CarlaAlpamayoAgent
from .sensor_manager import SensorManager
from .alpamayo_wrapper import AlpamayoWrapper

__all__ = [
    "CarlaAlpamayoAgent",
    "SensorManager",
    "AlpamayoWrapper",
]
