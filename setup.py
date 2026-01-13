#!/usr/bin/env python3
"""Setup script for CARLA-Alpamayo integration"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="carla-alpamayo",
    version="0.1.0",
    author="hwkim3330",
    description="NVIDIA Alpamayo VLA model integration with CARLA simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hwkim3330/carla-alpamayo",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "huggingface_hub>=0.20.0",
    ],
    extras_require={
        "viz": [
            "matplotlib>=3.5.0",
            "opencv-python>=4.5.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "carla-alpamayo=examples.run_agent:main",
        ],
    },
)
