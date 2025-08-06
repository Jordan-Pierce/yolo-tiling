#!/usr/bin/env python

"""Tests for CLI functionality of yolo_tiler package."""

import subprocess
import sys
from pathlib import Path


def test_cli_basic_functionality():
    """Test basic CLI functionality with different annotation types"""
    
    test_cases = [
        {
            "name": "Object Detection",
            "args": [
                "--source", "./tests/detection",
                "--target", "./tests/detection_tiled_cli",
                "--annotation_type", "object_detection",
                "--slice_wh", "320", "240",
                "--input_ext", ".png"
            ]
        },
        {
            "name": "Image Classification", 
            "args": [
                "--source", "./tests/classification",
                "--target", "./tests/classification_tiled_cli",
                "--annotation_type", "image_classification",
                "--slice_wh", "320", "240",
                "--input_ext", ".jpg",
                "--copy_source_data"
            ]
        },
        {
            "name": "Instance Segmentation",
            "args": [
                "--source", "./tests/segmentation", 
                "--target", "./tests/segmentation_tiled_cli",
                "--annotation_type", "instance_segmentation",
                "--slice_wh", "320", "240",
                "--input_ext", ".png"
            ]
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting CLI: {test_case['name']}")
        try:
            # Create target directory if it doesn't exist
            target_dir = Path(test_case['args'][test_case['args'].index('--target') + 1])
            target_dir.mkdir(parents=True, exist_ok=True)
            
            result = subprocess.run([
                sys.executable, 
                "-m", 
                "yolo_tiler.cli"
            ] + test_case['args'], 
            capture_output=True, text=True, check=True, timeout=300)
            
            print(f"✓ {test_case['name']} CLI test passed")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ {test_case['name']} CLI test failed: {e}")
            print(f"Error output: {e.stderr}")
        except subprocess.TimeoutExpired:
            print(f"✗ {test_case['name']} CLI test timed out")


def test_cli_help():
    """Test CLI help functionality"""
    print("\nTesting CLI Help...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "yolo_tiler.cli", "--help"
        ], capture_output=True, text=True, check=True)
        
        if "usage:" in result.stdout.lower() and "yolo" in result.stdout.lower():
            print("✓ CLI help works correctly")
        else:
            print("✗ CLI help output seems incorrect")
            
    except Exception as e:
        print(f"✗ CLI help test failed: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("TESTING CLI FUNCTIONALITY")
    print("=" * 50)
    
    test_cli_help()
    test_cli_basic_functionality()
        
    print("\nCLI tests completed!")