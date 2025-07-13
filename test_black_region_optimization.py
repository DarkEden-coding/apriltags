#!/usr/bin/env python3
"""
Test script to verify black region optimization in AprilTag detector.
"""
import numpy as np
import pupil_apriltags as apriltags
import time

def create_test_image_with_black_regions():
    """Create a test image with large black regions and a simple pattern."""
    # Create a 640x480 image with large black regions
    image = np.zeros((480, 640), dtype=np.uint8)
    
    # Add some white areas and patterns (simulating where tags might be)
    # Top-left corner - white square
    image[50:150, 50:150] = 255
    
    # Center - white square with black border (simulating a tag)
    image[200:300, 270:370] = 255
    image[220:280, 290:350] = 0
    
    # Bottom-right corner - white square
    image[350:450, 490:590] = 255
    
    # Add some noise to make it more realistic
    noise = np.random.randint(0, 30, size=(480, 640), dtype=np.uint8)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image

def benchmark_detector(image, detector, test_name):
    """Benchmark the detector with the given image."""
    print(f"\n{test_name}:")
    
    # Warmup
    detector.detect(image)
    
    # Benchmark
    start_time = time.time()
    num_runs = 10
    for _ in range(num_runs):
        detections = detector.detect(image)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    print(f"  Average time per detection: {avg_time:.4f} seconds")
    print(f"  Tags detected: {len(detections)}")
    
    return avg_time

def test_black_region_optimization():
    """Test the black region optimization feature."""
    print("Testing Black Region Optimization for AprilTag Detection")
    print("=" * 60)
    
    # Create test image
    image = create_test_image_with_black_regions()
    
    # Calculate black region percentage
    black_pixels = np.sum(image <= 10)
    total_pixels = image.size
    black_percentage = black_pixels / total_pixels * 100
    print(f"Test image: {image.shape[1]}x{image.shape[0]} pixels")
    print(f"Black regions: {black_percentage:.1f}% of image")
    
    # Test with optimization disabled
    detector_no_opt = apriltags.Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )
    
    # Test with optimization enabled (default)
    detector_with_opt = apriltags.Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )
    
    # Benchmark both detectors
    time_no_opt = benchmark_detector(image, detector_no_opt, "Detector without black region optimization")
    time_with_opt = benchmark_detector(image, detector_with_opt, "Detector with black region optimization")
    
    # Calculate improvement
    if time_no_opt > 0:
        improvement = ((time_no_opt - time_with_opt) / time_no_opt) * 100
        print(f"\nPerformance improvement: {improvement:.1f}%")
    
    # Test with different black region parameters
    print(f"\nTesting with different black region parameters:")
    
    # More aggressive optimization
    detector_aggressive = apriltags.Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )
    
    time_aggressive = benchmark_detector(image, detector_aggressive, "Detector with aggressive optimization")
    
    print(f"\nSummary:")
    print(f"  Without optimization: {time_no_opt:.4f}s")
    print(f"  With optimization:    {time_with_opt:.4f}s")
    print(f"  Aggressive mode:      {time_aggressive:.4f}s")
    
    return time_no_opt, time_with_opt, time_aggressive

if __name__ == "__main__":
    test_black_region_optimization()