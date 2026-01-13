"""
Alpamayo Model Wrapper
Handles loading and inference of the NVIDIA Alpamayo VLA model
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image


@dataclass
class AlpamayoOutput:
    """Output from Alpamayo model inference"""
    steering: float  # -1.0 to 1.0
    throttle: float  # 0.0 to 1.0
    brake: float     # 0.0 to 1.0
    reasoning: str   # Chain-of-thought reasoning text
    trajectory: Optional[np.ndarray] = None  # Predicted waypoints


class AlpamayoWrapper:
    """
    Wrapper for NVIDIA Alpamayo-R1 VLA model
    Provides interface for autonomous driving inference
    """

    def __init__(
        self,
        model_name: str = "nvidia/Alpamayo-R1-10B",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        use_flash_attention: bool = True,
    ):
        """
        Initialize Alpamayo model wrapper

        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on
            dtype: Model precision
            use_flash_attention: Use Flash Attention 2 for efficiency
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention

        self.model = None
        self.processor = None
        self.is_loaded = False

    def load_model(self) -> None:
        """Load Alpamayo model from HuggingFace"""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            print(f"Loading Alpamayo model: {self.model_name}")
            print("This may take a few minutes (~22GB download)...")

            # Configure attention implementation
            attn_impl = "flash_attention_2" if self.use_flash_attention else "sdpa"

            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map=self.device,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            )

            self.model.eval()
            self.is_loaded = True
            print("Model loaded successfully!")

        except ImportError as e:
            raise ImportError(
                "Required packages not found. Install with:\n"
                "pip install transformers torch accelerate"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load Alpamayo model: {e}") from e

    def preprocess_image(self, image: np.ndarray) -> Image.Image:
        """
        Preprocess camera image for model input

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            Preprocessed PIL Image
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        return Image.fromarray(image)

    def create_prompt(
        self,
        navigation_command: str = "follow the road",
        speed_limit: float = 30.0,
        current_speed: float = 0.0,
    ) -> str:
        """
        Create input prompt for Alpamayo model

        Args:
            navigation_command: High-level navigation instruction
            speed_limit: Current speed limit in km/h
            current_speed: Current vehicle speed in km/h

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an autonomous vehicle AI assistant. Analyze the driving scene and provide safe driving actions.

Navigation: {navigation_command}
Speed Limit: {speed_limit:.1f} km/h
Current Speed: {current_speed:.1f} km/h

Analyze the scene step by step:
1. Identify road layout and lane markings
2. Detect vehicles, pedestrians, and obstacles
3. Assess traffic signals and signs
4. Determine safe trajectory

Provide your reasoning and then output control commands in the format:
STEERING: <value between -1.0 and 1.0>
THROTTLE: <value between 0.0 and 1.0>
BRAKE: <value between 0.0 and 1.0>
"""
        return prompt

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        navigation_command: str = "follow the road",
        speed_limit: float = 30.0,
        current_speed: float = 0.0,
        max_new_tokens: int = 512,
    ) -> AlpamayoOutput:
        """
        Run inference on camera image

        Args:
            image: RGB camera image (H, W, 3)
            navigation_command: Navigation instruction
            speed_limit: Speed limit in km/h
            current_speed: Current speed in km/h
            max_new_tokens: Maximum tokens to generate

        Returns:
            AlpamayoOutput with control commands and reasoning
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Preprocess image
        pil_image = self.preprocess_image(image)

        # Create prompt
        prompt = self.create_prompt(navigation_command, speed_limit, current_speed)

        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt",
        ).to(self.device)

        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # Decode response
        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Parse control commands from response
        return self._parse_response(response)

    def _parse_response(self, response: str) -> AlpamayoOutput:
        """Parse model response to extract control commands"""
        import re

        # Default values
        steering = 0.0
        throttle = 0.0
        brake = 0.0

        # Extract values using regex
        steering_match = re.search(r'STEERING:\s*([-\d.]+)', response)
        throttle_match = re.search(r'THROTTLE:\s*([\d.]+)', response)
        brake_match = re.search(r'BRAKE:\s*([\d.]+)', response)

        if steering_match:
            steering = float(steering_match.group(1))
            steering = max(-1.0, min(1.0, steering))

        if throttle_match:
            throttle = float(throttle_match.group(1))
            throttle = max(0.0, min(1.0, throttle))

        if brake_match:
            brake = float(brake_match.group(1))
            brake = max(0.0, min(1.0, brake))

        return AlpamayoOutput(
            steering=steering,
            throttle=throttle,
            brake=brake,
            reasoning=response,
        )

    def predict_dummy(
        self,
        image: np.ndarray,
        **kwargs,
    ) -> AlpamayoOutput:
        """
        Dummy prediction for testing without model
        Returns simple lane-following behavior
        """
        # Simple dummy logic - drive straight slowly
        return AlpamayoOutput(
            steering=0.0,
            throttle=0.3,
            brake=0.0,
            reasoning="[DUMMY MODE] Driving straight at low speed.",
        )
