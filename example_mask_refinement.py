#!/usr/bin/env python3
"""
Example script demonstrating mask refinement usage.
This shows different ways to use the MaskRefiner class.
"""

import numpy as np
import os
from PIL import Image
from mask_refiner import MaskRefiner

def example_basic_usage():
    """Basic usage example with default parameters."""
    print("=== Basic Usage Example ===")
    
    # Initialize with default parameters (good for most cases)
    refiner = MaskRefiner()
    
    # Process a single mask file
    input_mask_path = "segmentation_masks/full_frame_0050_mask.png"
    if os.path.exists(input_mask_path):
        refined_mask = refiner.refine_mask(input_mask_path)
        
        # Save result
        output_path = "example_refined_mask.png"
        Image.fromarray(refined_mask).save(output_path)
        print(f"Refined mask saved to: {output_path}")
    else:
        print(f"Input mask not found: {input_mask_path}")

def example_custom_parameters():
    """Example with custom parameters for aggressive cleaning."""
    print("\n=== Custom Parameters Example ===")
    
    # For very noisy masks that need aggressive cleaning
    aggressive_refiner = MaskRefiner(
        min_object_size=200,           # Remove smaller objects
        hole_fill_area_threshold=100,  # Fill larger holes
        morphology_kernel_size=5,      # Stronger smoothing
        gaussian_sigma=1.0,            # More aggressive smoothing
        preserve_largest_component=True # Keep only main object
    )
    
    # For subtle cleaning that preserves details
    gentle_refiner = MaskRefiner(
        min_object_size=50,            # Keep smaller details
        hole_fill_area_threshold=20,   # Fill only small holes
        morphology_kernel_size=2,      # Minimal smoothing
        gaussian_sigma=0.3,            # Light smoothing
        preserve_largest_component=False # Keep multiple objects
    )
    
    input_mask_path = "segmentation_masks/full_frame_0050_mask.png"
    if os.path.exists(input_mask_path):
        # Process with both refiners
        aggressive_result = aggressive_refiner.refine_mask(input_mask_path)
        gentle_result = gentle_refiner.refine_mask(input_mask_path)
        
        # Save results
        Image.fromarray(aggressive_result).save("example_aggressive_refined.png")
        Image.fromarray(gentle_result).save("example_gentle_refined.png")
        print("Saved aggressive_refined.png and gentle_refined.png")

def example_batch_processing():
    """Example of batch processing multiple masks."""
    print("\n=== Batch Processing Example ===")
    
    # Initialize refiner optimized for bladder segmentation
    bladder_refiner = MaskRefiner(
        min_object_size=150,           # Remove small noise typical in medical images
        hole_fill_area_threshold=75,   # Fill medium-sized holes
        morphology_kernel_size=3,      # Standard smoothing
        gaussian_sigma=0.5,            # Light final smoothing
        preserve_largest_component=True # Keep only main bladder region
    )
    
    # Process all masks in the segmentation_masks directory
    input_dir = "segmentation_masks"
    output_dir = "refined_masks"
    
    if os.path.exists(input_dir):
        print(f"Processing masks from {input_dir}...")
        bladder_refiner.batch_refine(
            input_dir=input_dir,
            output_dir=output_dir,
            suffix="_refined",
            verbose=True
        )
        print(f"All refined masks saved to {output_dir}")
    else:
        print(f"Input directory not found: {input_dir}")

def example_with_probability_maps():
    """Example using probability maps with adaptive thresholding."""
    print("\n=== Probability Map Example ===")
    
    # Create a mock probability map (in practice, this would come from your model)
    # This simulates a soft segmentation output
    height, width = 256, 256
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    
    # Create a soft circular mask with gradient
    radius = 80
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    probability_map = np.clip(1.0 - distance / radius, 0, 1)
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.1, probability_map.shape)
    probability_map = np.clip(probability_map + noise, 0, 1)
    
    # Initialize refiner with adaptive thresholding
    adaptive_refiner = MaskRefiner(
        use_adaptive_threshold=True,
        min_object_size=100,
        preserve_largest_component=True
    )
    
    # Refine using probability map
    refined_mask = adaptive_refiner.refine_mask(
        input_mask=None,  # Not used when probability_map is provided
        probability_map=probability_map
    )
    
    # Save probability map and result for comparison
    prob_image = (probability_map * 255).astype(np.uint8)
    Image.fromarray(prob_image).save("example_probability_map.png")
    Image.fromarray(refined_mask).save("example_adaptive_refined.png")
    print("Saved probability_map.png and adaptive_refined.png")

def example_integration_workflow():
    """Example showing how to integrate with existing SAMUS workflow."""
    print("\n=== Integration Workflow Example ===")
    
    # This is how you might modify your inference pipeline
    def refined_inference_pipeline(image_path, model, refiner):
        """
        Example pipeline that includes mask refinement.
        
        Args:
            image_path: Path to input image
            model: Your trained SAMUS model
            refiner: MaskRefiner instance
        """
        # Step 1: Run SAMUS inference (pseudo-code)
        # raw_prediction = model.predict(image_path)
        # probability_map = torch.sigmoid(raw_prediction).cpu().numpy()
        
        # For this example, we'll simulate this step
        print(f"Processing {image_path} with SAMUS + Refinement")
        
        # Step 2: Apply refinement
        # refined_mask = refiner.refine_mask(
        #     input_mask=None,
        #     probability_map=probability_map
        # )
        
        # For demo, just load an existing mask and refine it
        if os.path.exists("segmentation_masks/full_frame_0050_mask.png"):
            refined_mask = refiner.refine_mask("segmentation_masks/full_frame_0050_mask.png")
            return refined_mask
        
        return None
    
    # Initialize refiner for your specific use case
    production_refiner = MaskRefiner(
        min_object_size=100,
        hole_fill_area_threshold=50,
        morphology_kernel_size=3,
        gaussian_sigma=0.5,
        preserve_largest_component=True
    )
    
    # Example usage
    result = refined_inference_pipeline("dummy_image.png", None, production_refiner)
    if result is not None:
        Image.fromarray(result).save("example_integrated_result.png")
        print("Saved integrated_result.png")

def print_performance_tips():
    """Print performance optimization tips."""
    print("\n=== Performance Tips ===")
    print("1. For batch processing, use larger min_object_size to speed up processing")
    print("2. Reduce gaussian_sigma or set to 0 for faster processing")
    print("3. Set preserve_largest_component=True for single-object segmentation")
    print("4. Use smaller morphology_kernel_size for faster morphological operations")
    print("5. Process images in batches rather than one-by-one for better efficiency")
    print("6. Consider processing at lower resolution and upsampling if speed is critical")

def main():
    """Run all examples."""
    print("Mask Refinement Examples")
    print("=" * 50)
    
    # Create output directory for examples
    os.makedirs("example_outputs", exist_ok=True)
    os.chdir("example_outputs")
    
    try:
        # Run examples
        example_basic_usage()
        example_custom_parameters()
        example_batch_processing()
        example_with_probability_maps()
        example_integration_workflow()
        print_performance_tips()
        
        print("\n=== All Examples Complete ===")
        print("Check the 'example_outputs' directory for results")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have some mask files in the segmentation_masks directory")

if __name__ == "__main__":
    main() 