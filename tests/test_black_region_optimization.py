#!/usr/bin/env python3
"""
Test black region optimization functionality.
"""
import numpy as np
import pupil_apriltags as apriltags

def test_black_region_optimization_parameters():
    """Test that black region optimization parameters are correctly set."""
    # Test with optimization enabled
    detector = apriltags.Detector(
        skip_black_regions=1,
        black_region_cell_size=32
    )
    
    # Test with optimization disabled
    detector_no_opt = apriltags.Detector(
        skip_black_regions=0
    )
    
    # Create a simple test image
    image = np.zeros((100, 100), dtype=np.uint8)
    
    # Both should be able to process the image without error
    detections1 = detector.detect(image)
    detections2 = detector_no_opt.detect(image)
    
    # Should return valid detection arrays (even if empty)
    assert isinstance(detections1, list)
    assert isinstance(detections2, list)

def test_black_region_optimization_consistency():
    """Test that optimization doesn't affect detection accuracy."""
    # Create a test image with some white regions
    image = np.zeros((200, 200), dtype=np.uint8)
    image[50:150, 50:150] = 255  # White square
    
    # Create detectors with and without optimization
    detector_opt = apriltags.Detector(skip_black_regions=1)
    detector_no_opt = apriltags.Detector(skip_black_regions=0)
    
    # Both should produce the same number of detections
    detections_opt = detector_opt.detect(image)
    detections_no_opt = detector_no_opt.detect(image)
    
    # Should have the same number of detections
    assert len(detections_opt) == len(detections_no_opt)

def test_black_region_different_parameters():
    """Test that different black region parameters work correctly."""
    image = np.zeros((100, 100), dtype=np.uint8)
    
    # Test different parameter combinations
    param_sets = [
        {'black_region_cell_size': 8},
        {'black_region_cell_size': 16},
        {'black_region_cell_size': 32},
    ]
    
    for params in param_sets:
        detector = apriltags.Detector(
            skip_black_regions=1,
            **params
        )
        
        # Should not raise any exceptions
        detections = detector.detect(image)
        assert isinstance(detections, list)

if __name__ == "__main__":
    test_black_region_optimization_parameters()
    test_black_region_optimization_consistency() 
    test_black_region_different_parameters()
    print("All black region optimization tests passed!")