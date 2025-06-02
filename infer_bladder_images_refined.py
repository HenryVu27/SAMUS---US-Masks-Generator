import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.model_dict import get_model
from utils.bladder_config import get_bladder_config
from mask_refiner import MaskRefiner

class BladderDataset(Dataset):
    def __init__(self, data_dir, img_size=256):
        self.data_dir = data_dir
        self.img_size = img_size
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        # Load and preprocess image
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        orig_size = image.size
        
        # Apply transformations
        image_tensor = self.transform(image)  # [1, H, W]
        
        # Convert grayscale to RGB
        image_tensor = image_tensor.repeat(3, 1, 1)  # [3, H, W]
        
        # Create point prompt at center
        prompt_x = self.img_size // 2
        prompt_y = self.img_size // 2
        
        # Create point coordinates and labels
        point_coords = torch.tensor([[prompt_x, prompt_y]], dtype=torch.float32)  # [1, 2]
        point_labels = torch.tensor([1], dtype=torch.float32)  # [1]
        
        return {
            'image': image_tensor,  # [3, H, W]
            'point_coords': point_coords,  # [1, 2]
            'point_labels': point_labels,  # [1]
            'image_name': img_name,
            'original_size': orig_size
        }

def generate_refined_masks(data_dir, 
                          output_dir, 
                          checkpoint_path='SAMUS.pth',
                          refinement_config=None,
                          save_probability_maps=False,
                          save_unrefined_masks=False):
    """
    Generate segmentation masks with integrated refinement.
    
    Args:
        data_dir: Directory containing input images
        output_dir: Directory to save refined masks
        checkpoint_path: Path to SAMUS model checkpoint
        refinement_config: Dictionary with refinement parameters
        save_probability_maps: Whether to save raw probability maps
        save_unrefined_masks: Whether to save masks before refinement
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    if save_probability_maps:
        os.makedirs(os.path.join(output_dir, 'probability_maps'), exist_ok=True)
    if save_unrefined_masks:
        os.makedirs(os.path.join(output_dir, 'unrefined'), exist_ok=True)
    
    # Initialize dataset and dataloader
    dataset = BladderDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Set up model
    opt = get_bladder_config()
    opt.load_path = checkpoint_path
    opt.mode = "val"
    opt.classes = 2
    opt.batch_size = 1
    opt.sam_ckpt = checkpoint_path
    
    # Add required model configuration
    opt.encoder_input_size = 256
    opt.vit_name = 'vit_b'
    opt.low_image_size = 128
    
    print(f"Loading model from: {checkpoint_path}")
    model = get_model('SAMUS', args=opt, opt=opt)
    model.to(device)
    model.eval()
    
    # Initialize mask refiner with custom config or defaults
    if refinement_config is None:
        refinement_config = {
            'min_object_size': 150,
            'hole_fill_area_threshold': 75,
            'morphology_kernel_size': 3,
            'gaussian_sigma': 0.5,
            'use_adaptive_threshold': True,
            'preserve_largest_component': True
        }
    
    refiner = MaskRefiner(**refinement_config)
    print(f"Initialized mask refiner with config: {refinement_config}")
    
    print(f"Processing {len(dataset)} images...")
    
    # Process each image
    refined_count = 0
    for i, batch in enumerate(dataloader):
        img_name = batch['image_name'][0]
        print(f"Processing image {i+1}/{len(dataset)}: {img_name}")
        
        # Move data to device
        images = batch['image'].to(device)  # [1, 3, H, W]
        point_coords = batch['point_coords'].to(device)  # [1, 1, 2]
        point_labels = batch['point_labels'].to(device)  # [1, 1]
        
        try:
            with torch.no_grad():
                pred = model(images, (point_coords, point_labels))
                
                # Get the mask - handle the output shape correctly
                if isinstance(pred, dict) and 'masks' in pred:
                    mask = pred['masks']
                    
                    # Apply sigmoid to get probability map
                    probability_map = torch.sigmoid(mask)
                    
                    # Convert to numpy and get the mask
                    prob_map_np = probability_map[0, 0].cpu().numpy()  # [H, W]
                    
                    # Get original size and convert to integers
                    orig_size = batch['original_size']
                    orig_width = int(orig_size[0].item())
                    orig_height = int(orig_size[1].item())
                    
                    # Resize probability map to original size
                    prob_map_resized = np.array(Image.fromarray(prob_map_np).resize(
                        (orig_width, orig_height), Image.BILINEAR
                    ))
                    
                    # Apply refinement using probability map
                    refined_mask = refiner.refine_mask(
                        input_mask=None,
                        probability_map=prob_map_resized
                    )
                    
                    # Create unrefined binary mask for comparison
                    if save_unrefined_masks:
                        unrefined_mask = (prob_map_resized > 0.5).astype(np.uint8) * 255
                        unrefined_mask_resized = Image.fromarray(unrefined_mask).resize(
                            (orig_width, orig_height), Image.NEAREST
                        )
                        
                        # Save unrefined mask
                        unrefined_name = os.path.splitext(img_name)[0] + '_unrefined.png'
                        unrefined_path = os.path.join(output_dir, 'unrefined', unrefined_name)
                        unrefined_mask_resized.save(unrefined_path)
                    
                    # Save probability map if requested
                    if save_probability_maps:
                        prob_map_8bit = (prob_map_resized * 255).astype(np.uint8)
                        prob_name = os.path.splitext(img_name)[0] + '_prob.png'
                        prob_path = os.path.join(output_dir, 'probability_maps', prob_name)
                        Image.fromarray(prob_map_8bit).save(prob_path)
                    
                    # Save the refined segmentation mask
                    output_name = os.path.splitext(img_name)[0] + '_mask_refined.png'
                    output_path = os.path.join(output_dir, output_name)
                    Image.fromarray(refined_mask).save(output_path)
                    
                    refined_count += 1
                    print(f"Saved refined mask to: {output_path}")
                    
                else:
                    raise ValueError(f"Unexpected model output format: {type(pred)}")
            
        except Exception as e:
            print(f"Error processing image {img_name}: {str(e)}")
            print(f"Full error details: {type(e).__name__}: {str(e)}")
            continue
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {refined_count}/{len(dataset)} images")
    print(f"Refined segmentation masks saved to {output_dir}")

def create_refinement_comparison(original_dir, refined_dir, sample_count=5):
    """
    Create a visual comparison between original and refined masks.
    
    Args:
        original_dir: Directory with original masks
        refined_dir: Directory with refined masks
        sample_count: Number of samples to compare
    """
    try:
        import matplotlib.pyplot as plt
        
        # Find matching files
        orig_files = [f for f in os.listdir(original_dir) if f.endswith('_mask.png')]
        refined_files = [f for f in os.listdir(refined_dir) if f.endswith('_mask_refined.png')]
        
        # Match original and refined files
        matches = []
        for orig_file in orig_files[:sample_count]:
            base_name = orig_file.replace('_mask.png', '')
            refined_file = base_name + '_mask_refined.png'
            if refined_file in refined_files:
                matches.append((orig_file, refined_file))
        
        if not matches:
            print("No matching files found for comparison")
            return
        
        # Create comparison plot
        fig, axes = plt.subplots(len(matches), 2, figsize=(10, 3 * len(matches)))
        if len(matches) == 1:
            axes = axes.reshape(1, -1)
        
        for i, (orig_file, refined_file) in enumerate(matches):
            # Load masks
            orig_path = os.path.join(original_dir, orig_file)
            refined_path = os.path.join(refined_dir, refined_file)
            
            orig_mask = np.array(Image.open(orig_path).convert('L'))
            refined_mask = np.array(Image.open(refined_path).convert('L'))
            
            # Plot original
            axes[i, 0].imshow(orig_mask, cmap='gray', vmin=0, vmax=255)
            axes[i, 0].set_title(f'Original: {orig_file}')
            axes[i, 0].axis('off')
            
            # Plot refined
            axes[i, 1].imshow(refined_mask, cmap='gray', vmin=0, vmax=255)
            axes[i, 1].set_title(f'Refined: {refined_file}')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        comparison_path = os.path.join(refined_dir, 'refinement_comparison.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison saved to: {comparison_path}")
        
    except ImportError:
        print("matplotlib not available, skipping comparison visualization")
    except Exception as e:
        print(f"Error creating comparison: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate refined segmentation masks for bladder images')
    parser.add_argument('--data_dir', default='bladder', help='Directory containing input images')
    parser.add_argument('--output_dir', default='refined_segmentation_masks', help='Directory to save refined masks')
    parser.add_argument('--checkpoint', default='SAMUS.pth', help='Path to model checkpoint')
    
    # Refinement parameters
    parser.add_argument('--min_object_size', type=int, default=150, 
                       help='Minimum object size to keep')
    parser.add_argument('--hole_fill_threshold', type=int, default=75,
                       help='Maximum hole size to fill')
    parser.add_argument('--kernel_size', type=int, default=3,
                       help='Morphological kernel size')
    parser.add_argument('--gaussian_sigma', type=float, default=0.5,
                       help='Gaussian smoothing sigma')
    parser.add_argument('--preserve_largest', action='store_true', default=True,
                       help='Keep only largest connected component')
    parser.add_argument('--adaptive_threshold', action='store_true', default=True,
                       help='Use adaptive thresholding')
    
    # Output options
    parser.add_argument('--save_probability_maps', action='store_true',
                       help='Save raw probability maps')
    parser.add_argument('--save_unrefined', action='store_true',
                       help='Save unrefined masks for comparison')
    parser.add_argument('--create_comparison', action='store_true',
                       help='Create before/after comparison visualization')
    
    args = parser.parse_args()
    
    # Build refinement configuration
    refinement_config = {
        'min_object_size': args.min_object_size,
        'hole_fill_area_threshold': args.hole_fill_threshold,
        'morphology_kernel_size': args.kernel_size,
        'gaussian_sigma': args.gaussian_sigma,
        'use_adaptive_threshold': args.adaptive_threshold,
        'preserve_largest_component': args.preserve_largest
    }
    
    print("Refined SAMUS Inference Pipeline")
    print("=" * 40)
    print(f"Input directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Refinement config: {refinement_config}")
    print("=" * 40)
    
    # Run inference with refinement
    generate_refined_masks(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        refinement_config=refinement_config,
        save_probability_maps=args.save_probability_maps,
        save_unrefined_masks=args.save_unrefined
    )
    
    # Create comparison if requested
    if args.create_comparison:
        original_dir = 'segmentation_masks'  # Your existing masks
        create_refinement_comparison(original_dir, args.output_dir) 