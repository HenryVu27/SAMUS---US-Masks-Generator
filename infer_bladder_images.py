import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.model_dict import get_model
from utils.bladder_config import get_bladder_config

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

def generate_masks(data_dir, output_dir, checkpoint_path='SAMUS.pth'):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    print(f"Processing {len(dataset)} images...")
    
    # Process each image
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
                print(f"Model output keys: {pred.keys()}")
                print(f"Prediction type: {type(pred)}")
                
                # Get the mask - handle the output shape correctly
                if isinstance(pred, dict) and 'masks' in pred:
                    mask = pred['masks']
                    print(f"Raw mask shape: {mask.shape}")
                    mask = torch.sigmoid(mask)
                    print(f"After sigmoid shape: {mask.shape}")
                    
                    # Convert to numpy and get the mask
                    mask = mask[0, 0].cpu().numpy()  # Take first batch and first channel
                    print(f"Final mask shape: {mask.shape}")
                    
                    # Create binary mask
                    binary_mask = (mask > 0.5).astype(np.uint8) * 255
                    
                    # Get original size and convert to integers
                    orig_size = batch['original_size']
                    orig_width = int(orig_size[0].item())
                    orig_height = int(orig_size[1].item())
                    print(f"Original size: {orig_width}x{orig_height}")
                    
                    # Resize to original size
                    binary_mask = Image.fromarray(binary_mask).resize(
                        (orig_width, orig_height),
                        Image.NEAREST
                    )
                    
                    # Save the segmentation mask
                    output_name = os.path.splitext(img_name)[0] + '_mask.png'
                    output_path = os.path.join(output_dir, output_name)
                    binary_mask.save(output_path)
                    print(f"Saved mask to: {output_path}")
                else:
                    raise ValueError(f"Unexpected model output format: {type(pred)}")
            
        except Exception as e:
            print(f"Error processing image {img_name}: {str(e)}")
            print(f"Full error details: {type(e).__name__}: {str(e)}")
            continue
    
    print(f"Segmentation masks saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate segmentation masks for bladder images')
    parser.add_argument('--data_dir', default='bladder', help='Directory containing input images')
    parser.add_argument('--output_dir', default='segmentation_masks', help='Directory to save masks')
    parser.add_argument('--checkpoint', default='SAMUS.pth', help='Path to model checkpoint')
    
    args = parser.parse_args()
    generate_masks(args.data_dir, args.output_dir, args.checkpoint) 