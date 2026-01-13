#!/usr/bin/env python3
"""
Download Alpamayo model from HuggingFace
Run this script before using the agent with the full model
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Download Alpamayo model")
    parser.add_argument("--model", default="nvidia/Alpamayo-R1-10B",
                        help="Model name on HuggingFace")
    parser.add_argument("--cache-dir", default=None,
                        help="Custom cache directory")
    args = parser.parse_args()

    print("=" * 60)
    print("NVIDIA Alpamayo Model Download")
    print("=" * 60)
    print()
    print("Prerequisites:")
    print("1. Request access to the model at:")
    print(f"   https://huggingface.co/{args.model}")
    print()
    print("2. Login to HuggingFace:")
    print("   huggingface-cli login")
    print()
    print("Note: Model size is approximately 22GB")
    print("=" * 60)

    response = input("\nProceed with download? [y/N]: ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return

    try:
        from huggingface_hub import snapshot_download

        print(f"\nDownloading {args.model}...")
        print("This may take 10-30 minutes depending on your connection.")

        path = snapshot_download(
            repo_id=args.model,
            cache_dir=args.cache_dir,
            resume_download=True,
        )

        print(f"\nModel downloaded to: {path}")
        print("You can now run the agent without --dummy flag.")

    except ImportError:
        print("\nError: huggingface_hub not installed.")
        print("Install with: pip install huggingface_hub")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your HuggingFace login: huggingface-cli whoami")
        print("2. Ensure you have access to the model")
        print("3. Check your internet connection")


if __name__ == "__main__":
    main()
