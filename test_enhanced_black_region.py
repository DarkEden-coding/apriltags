#!/usr/bin/env python3
"""
Enhanced test script to verify black region optimization in AprilTag detector.
"""
import numpy as np
import pupil_apriltags as apriltags
import time

def create_test_image_with_mostly_black():
    """Create a test image with 70% black regions and some white patterns."""
    # Create a 640x480 image with mostly black regions
    image = np.zeros((480, 640), dtype=np.uint8)
    
    # Add some white areas in small regions (simulating where tags might be)
    # Top-left corner - small white square
    image[50:90, 50:90] = 255
    
    # Center area - white square with black border (simulating a tag)
    image[220:260, 300:340] = 255
    image[230:250, 310:330] = 0
    
    # Bottom-right corner - small white square
    image[400:440, 550:590] = 255
    
    # Add some random noise in small areas
    for _ in range(10):
        y = np.random.randint(0, 400)
        x = np.random.randint(0, 600)
        image[y:y+20, x:x+20] = np.random.randint(200, 255)
    
    return image

def test_black_region_optimization_enhanced():
    """Test the black region optimization feature with more comprehensive settings."""
    print("Enhanced Black Region Optimization Test")
    print("=" * 50)
    
    # Create test image with mostly black regions
    image = create_test_image_with_mostly_black()
    
    # Calculate black region percentage
    black_pixels = np.sum(image <= 10)
    total_pixels = image.size
    black_percentage = black_pixels / total_pixels * 100
    print(f"Test image: {image.shape[1]}x{image.shape[0]} pixels")
    print(f"Black regions: {black_percentage:.1f}% of image")
    
    # Test configurations
    configs = [
        {
            'name': 'No optimization',
            'params': {
                'skip_black_regions': 0
            }
        },
        {
            'name': 'Default optimization',
            'params': {
                'skip_black_regions': 1,
                'black_region_cell_size': 16,
                'black_region_threshold': 10,
                'black_region_percentage': 0.95
            }
        },
        {
            'name': 'Aggressive optimization',
            'params': {
                'skip_black_regions': 1,
                'black_region_cell_size': 32,
                'black_region_threshold': 20,
                'black_region_percentage': 0.8
            }
        },
        {
            'name': 'Conservative optimization',
            'params': {
                'skip_black_regions': 1,
                'black_region_cell_size': 8,
                'black_region_threshold': 5,
                'black_region_percentage': 0.99
            }
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{config['name']}:")
        
        # Create detector with specific configuration
        detector = apriltags.Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
            **config['params']
        )
        
        # Warmup
        detector.detect(image)
        
        # Benchmark
        start_time = time.time()
        num_runs = 20
        for _ in range(num_runs):
            detections = detector.detect(image)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        print(f"  Average time per detection: {avg_time:.4f} seconds")
        print(f"  Tags detected: {len(detections)}")
        
        results.append({
            'name': config['name'],
            'time': avg_time,
            'detections': len(detections)
        })
    
    # Print comparison
    print(f"\n{'Configuration':<25} {'Time (s)':<10} {'Speedup':<10} {'Detections':<10}")
    print("-" * 60)
    
    baseline_time = results[0]['time']
    for result in results:
        speedup = f"{(baseline_time / result['time']):.2f}x" if result['time'] > 0 else "N/A"
        print(f"{result['name']:<25} {result['time']:<10.4f} {speedup:<10} {result['detections']:<10}")
    
    # Test with different image sizes
    print(f"\nTesting with different image sizes:")
    print("-" * 40)
    
    sizes = [(320, 240), (640, 480), (1280, 720)]
    for width, height in sizes:
        # Create image with black regions
        test_image = np.zeros((height, width), dtype=np.uint8)
        # Add a few white spots
        test_image[height//4:height//4+20, width//4:width//4+20] = 255
        test_image[3*height//4:3*height//4+20, 3*width//4:3*width//4+20] = 255
        
        # Test with and without optimization
        detector_no_opt = apriltags.Detector(skip_black_regions=0)
        detector_opt = apriltags.Detector(skip_black_regions=1)
        
        # Benchmark
        start = time.time()
        for _ in range(5):
            detector_no_opt.detect(test_image)
        time_no_opt = (time.time() - start) / 5
        
        start = time.time()
        for _ in range(5):
            detector_opt.detect(test_image)
        time_opt = (time.time() - start) / 5
        
        improvement = ((time_no_opt - time_opt) / time_no_opt) * 100 if time_no_opt > 0 else 0
        
        print(f"  {width}x{height}: {time_no_opt:.4f}s -> {time_opt:.4f}s ({improvement:.1f}% improvement)")

    return results

if __name__ == "__main__":
    test_black_region_optimization_enhanced()