import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology, measure, filters
from PIL import Image
import os
import argparse
import time
from typing import Union, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class MaskRefiner:

    def __init__(self, 
                 min_object_size: int = 100,
                 hole_fill_area_threshold: int = 50,
                 morphology_kernel_size: int = 3,
                 gaussian_sigma: float = 0.5,
                 use_adaptive_threshold: bool = True,
                 preserve_largest_component: bool = True):

        self.min_object_size = min_object_size
        self.hole_fill_area_threshold = hole_fill_area_threshold
        self.morphology_kernel_size = morphology_kernel_size
        self.gaussian_sigma = gaussian_sigma
        self.use_adaptive_threshold = use_adaptive_threshold
        self.preserve_largest_component = preserve_largest_component
        
        # Create morphological kernels
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (morphology_kernel_size, morphology_kernel_size)
        )
        
    def adaptive_threshold(self, probability_map: np.ndarray) -> float:

        prob_8bit = (probability_map * 255).astype(np.uint8)
        
        # Apply Gaussian blur to reduce noise before thresholding
        prob_8bit = cv2.GaussianBlur(prob_8bit, (5, 5), 0)
        
        # Compute Otsu threshold
        threshold_val, _ = cv2.threshold(prob_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return threshold_val / 255.0
    
    def remove_small_objects(self, binary_mask: np.ndarray) -> np.ndarray:

        mask_bool = binary_mask > 0
        
        # Remove small objects
        cleaned = morphology.remove_small_objects(
            mask_bool, 
            min_size=self.min_object_size,
            connectivity=2
        )
        
        return (cleaned * 255).astype(np.uint8)
    
    def fill_holes(self, binary_mask: np.ndarray) -> np.ndarray:

        mask_bool = binary_mask > 0
        
        # Fill holes
        filled = ndimage.binary_fill_holes(mask_bool)
        
        # If we want to be more selective about hole filling
        if self.hole_fill_area_threshold > 0:
            # Find holes and filter by size
            holes = filled & ~mask_bool
            labeled_holes = measure.label(holes)
            
            for region in measure.regionprops(labeled_holes):
                if region.area > self.hole_fill_area_threshold:
                    # Don't fill large holes
                    filled[labeled_holes == region.label] = False
        
        return (filled * 255).astype(np.uint8)
    
    def morphological_operations(self, binary_mask: np.ndarray) -> np.ndarray:

        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, self.kernel)
        
        # Closing: dilation followed by erosion (fills small holes, connects nearby objects)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel)
        
        return closed
    
    def get_largest_component(self, binary_mask: np.ndarray) -> np.ndarray:

        num_labels, labels = cv2.connectedComponents(binary_mask)
        
        if num_labels <= 1:  # No objects found
            return binary_mask
        
        # Find largest component (excluding background label 0)
        largest_label = 1
        largest_size = 0
        
        for label in range(1, num_labels):
            size = np.sum(labels == label)
            if size > largest_size:
                largest_size = size
                largest_label = label
        
        # Create mask with only largest component
        largest_component = (labels == largest_label).astype(np.uint8) * 255
        
        return largest_component
    
    def gaussian_smoothing(self, binary_mask: np.ndarray) -> np.ndarray:

        if self.gaussian_sigma <= 0:
            return binary_mask
            
        # Convert to float for smoothing
        mask_float = binary_mask.astype(np.float32) / 255.0
        
        # Apply Gaussian filter
        smoothed = filters.gaussian(mask_float, sigma=self.gaussian_sigma)
        
        # Threshold back to binary
        smoothed_binary = (smoothed > 0.5).astype(np.uint8) * 255
        
        return smoothed_binary
    
    def refine_mask(self, 
                   input_mask: Union[np.ndarray, str], 
                   probability_map: Optional[np.ndarray] = None,
                   custom_threshold: Optional[float] = None) -> np.ndarray:

        # Load mask if it's a file path
        if isinstance(input_mask, str):
            mask = np.array(Image.open(input_mask).convert('L'))
        else:
            mask = input_mask.copy()
        
        # Handle probability maps vs binary masks
        if probability_map is not None:
            # Use probability map for thresholding
            if custom_threshold is not None:
                threshold = custom_threshold
            elif self.use_adaptive_threshold:
                threshold = self.adaptive_threshold(probability_map)
            else:
                threshold = 0.5
            
            mask = (probability_map > threshold).astype(np.uint8) * 255
        else:
            # Assume input is already binary or make it binary
            if mask.max() <= 1:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = (mask > 127).astype(np.uint8) * 255
        
        mask = self.remove_small_objects(mask)
        
        mask = self.fill_holes(mask)
        
        mask = self.morphological_operations(mask)
        
        if self.preserve_largest_component:
            mask = self.get_largest_component(mask)
        
        mask = self.gaussian_smoothing(mask)
        
        return mask
    
    def batch_refine(self, 
                    input_dir: str, 
                    output_dir: str,
                    suffix: str = "_refined",
                    verbose: bool = True) -> None:

        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        mask_files = [f for f in os.listdir(input_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        if verbose:
            print(f"Found {len(mask_files)} mask files to process")
        
        total_time = 0
        
        for i, filename in enumerate(mask_files):
            start_time = time.time()
            
            input_path = os.path.join(input_dir, filename)
            
            # Create output filename
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}{suffix}{ext}"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                # Refine mask
                refined_mask = self.refine_mask(input_path)
                
                # Save refined mask
                Image.fromarray(refined_mask).save(output_path)
                
                process_time = time.time() - start_time
                total_time += process_time
                
                if verbose:
                    print(f"[{i+1}/{len(mask_files)}] Processed {filename} -> {output_filename} "
                          f"({process_time:.3f}s)")
                    
            except Exception as e:
                if verbose:
                    print(f"Error processing {filename}: {str(e)}")
        
        if verbose:
            avg_time = total_time / len(mask_files) if mask_files else 0
            print(f"\nBatch processing complete!")
            print(f"Total time: {total_time:.2f}s")
            print(f"Average time per mask: {avg_time:.3f}s")


def create_comparison_grid(original_dir: str, 
                          refined_dir: str, 
                          output_path: str,
                          sample_files: Optional[list] = None,
                          max_samples: int = 4) -> None:
 
    if sample_files is None:
        # Get all matching files
        orig_files = set(os.listdir(original_dir))
        refined_files = set(os.listdir(refined_dir))
        
        # Find files with refined suffix
        matching_files = []
        for ref_file in refined_files:
            if ref_file.endswith('_refined.png'):
                orig_name = ref_file.replace('_refined.png', '_mask.png')
                if orig_name in orig_files:
                    matching_files.append((orig_name, ref_file))
        
        # Sample subset
        sample_files = matching_files[:max_samples]
    
    if not sample_files:
        print("No matching files found for comparison")
        return
    
    # Create comparison grid
    fig_height = len(sample_files) * 3
    fig, axes = plt.subplots(len(sample_files), 2, figsize=(8, fig_height))
    
    if len(sample_files) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (orig_file, refined_file) in enumerate(sample_files):
        # Load images
        orig_mask = np.array(Image.open(os.path.join(original_dir, orig_file)).convert('L'))
        refined_mask = np.array(Image.open(os.path.join(refined_dir, refined_file)).convert('L'))
        
        # Plot original
        axes[i, 0].imshow(orig_mask, cmap='gray')
        axes[i, 0].set_title(f'Original: {orig_file}')
        axes[i, 0].axis('off')
        
        # Plot refined
        axes[i, 1].imshow(refined_mask, cmap='gray')
        axes[i, 1].set_title(f'Refined: {refined_file}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison grid saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Refine segmentation masks')
    parser.add_argument('--input_dir', default='segmentation_masks', 
                       help='Directory containing input masks')
    parser.add_argument('--output_dir', default='refined_masks',
                       help='Directory to save refined masks')
    parser.add_argument('--min_object_size', type=int, default=100,
                       help='Minimum object size to keep')
    parser.add_argument('--hole_fill_threshold', type=int, default=50,
                       help='Maximum hole size to fill')
    parser.add_argument('--kernel_size', type=int, default=3,
                       help='Morphological kernel size')
    parser.add_argument('--gaussian_sigma', type=float, default=0.5,
                       help='Gaussian smoothing sigma')
    parser.add_argument('--preserve_largest', action='store_true',
                       help='Keep only largest connected component')
    parser.add_argument('--create_comparison', action='store_true',
                       help='Create before/after comparison grid')
    parser.add_argument('--single_file', type=str,
                       help='Process single file instead of batch')
    
    args = parser.parse_args()
    
    refiner = MaskRefiner(
        min_object_size=args.min_object_size,
        hole_fill_area_threshold=args.hole_fill_threshold,
        morphology_kernel_size=args.kernel_size,
        gaussian_sigma=args.gaussian_sigma,
        preserve_largest_component=args.preserve_largest
    )
    
    if args.single_file:
        # Process single file
        print(f"Processing single file: {args.single_file}")
        refined_mask = refiner.refine_mask(args.single_file)
        
        output_name = os.path.splitext(os.path.basename(args.single_file))[0] + "_refined.png"
        output_path = os.path.join(args.output_dir, output_name)
        os.makedirs(args.output_dir, exist_ok=True)
        Image.fromarray(refined_mask).save(output_path)
        print(f"Refined mask saved to: {output_path}")
        
    else:
        # Batch processing
        print(f"Batch processing masks from {args.input_dir}")
        refiner.batch_refine(args.input_dir, args.output_dir)
    
    if args.create_comparison:
        try:
            import matplotlib.pyplot as plt
            comparison_path = os.path.join(args.output_dir, "comparison_grid.png")
            create_comparison_grid(args.input_dir, args.output_dir, comparison_path)
        except ImportError:
            print("matplotlib not available, skipping comparison grid generation")


if __name__ == "__main__":
    main() 
