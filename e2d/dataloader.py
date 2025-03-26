import os
from typing import Tuple, Optional, Callable, List

import numpy as np
import torch
from math import sin, cos, pi
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image  # For perspective transformation

def calculate_valid_crop_size(angle_radians: float, width: int, height: int) -> Tuple[int, int]:
    """
    Calculate the largest valid crop size for an image with the given width and height
    after a rotation by angle_radians.
    """
    cos_angle = abs(cos(angle_radians))
    sin_angle = abs(sin(angle_radians))
    rotated_width = width * cos_angle + height * sin_angle
    rotated_height = width * sin_angle + height * cos_angle

    if rotated_width > 0 and rotated_height > 0:
        valid_crop_width = width * height / rotated_height
        valid_crop_height = width * height / rotated_width
    else:
        valid_crop_width = valid_crop_height = 0

    return int(np.floor(valid_crop_width)), int(np.floor(valid_crop_height))

def apply_vertical_motion_blur(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Apply vertical motion blur to a tensor image.
    Expects image to be a torch.Tensor of shape (C, H, W).
    """
    # Ensure kernel_size is odd to have a centered kernel.
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # If missing batch dimension, add one.
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Create a vertical blur kernel.
    kernel = torch.zeros((kernel_size, kernel_size), dtype=image.dtype, device=image.device)
    kernel[:, kernel_size // 2] = 1.0 / kernel_size
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    
    # Apply the kernel to each channel separately.
    if image.shape[1] > 1:
        blurred = torch.cat([
            torch.nn.functional.conv2d(image[:, i:i+1], kernel, padding='same')
            for i in range(image.shape[1])
        ], dim=1)
    else:
        blurred = torch.nn.functional.conv2d(image, kernel, padding='same')
    
    return blurred.squeeze(0)

def blur_images_with_sinusoidal_amplitude(image: torch.Tensor, max_kernel_size: int = 15, frequency: float = 0.1) -> torch.Tensor:
    """
    Apply vertical motion blur to the image with a kernel size modulated by a sinusoidal function.
    """
    amplitude = (np.sin(2 * np.pi * frequency * np.random.randint(0, 100)) + 1) / 2  
    kernel_size = int(1 + amplitude * (max_kernel_size - 1))
    kernel_size = max(1, kernel_size)
    return apply_vertical_motion_blur(image, kernel_size)

def augment_event(img: torch.Tensor,
                  rot_prob: float = 0.1,
                  blur_prob: float = 0.1,
                  intensity_prob: float = 0.2,
                  polarity_prob: float = 0.1,
                  crop_trans_prob: float = 0.1,
                  perspective_prob: float = 0.1) -> torch.Tensor:
    """
    Apply a series of augmentations to a single event tensor.
    Expects img to be a torch.Tensor of shape [C, H, W].
    """
    # Random rotation (with valid cropping and resizing).
    if np.random.rand() < rot_prob:
        angle = np.random.uniform(-20.0, 20.0)
        img = TF.rotate(img, angle)
        # Calculate new crop dimensions.
        new_w, new_h = calculate_valid_crop_size(angle * pi / 180, img.shape[-1], img.shape[-2])
        img = TF.resized_crop(
            img,
            top=img.shape[-2] // 2 - new_h // 2,
            left=img.shape[-1] // 2 - new_w // 2,
            height=new_h,
            width=new_w,
            size=(img.shape[-2], img.shape[-1])
        )
    
    # Apply vertical motion blur.
    if np.random.rand() < blur_prob:
        img = blur_images_with_sinusoidal_amplitude(img, max_kernel_size=15, frequency=0.1)
        
    # Intensity scaling to simulate varying illumination.
    if np.random.rand() < intensity_prob:
        scale_factor = np.random.uniform(0.25, 4.0)
        img = torch.clamp(img * scale_factor, 0.0, 1.0)
    
    # Polarity flipping can help with contrast variation.
    if np.random.rand() < polarity_prob:
        polarity = np.random.choice([-1.0, 1.0])
        img = img * polarity

    # Random cropping and translation.
    if np.random.rand() < crop_trans_prob:
        # Define maximum translation (10% of dimensions).
        max_dx = 0.1 * img.shape[-1]
        max_dy = 0.1 * img.shape[-2]
        tx = np.random.uniform(-max_dx, max_dx)
        ty = np.random.uniform(-max_dy, max_dy)
        img = TF.affine(img, angle=0, translate=[tx, ty], scale=1.0, shear=0)

    # Random perspective transformation.
    if np.random.rand() < perspective_prob:
        # Convert tensor to PIL Image.
        pil_img = TF.to_pil_image(img)
        width, height = pil_img.size
        startpoints = [[0, 0], [width, 0], [width, height], [0, height]]
        max_shift_x = 0.1 * width
        max_shift_y = 0.1 * height
        endpoints = [[pt[0] + np.random.uniform(-max_shift_x, max_shift_x),
                      pt[1] + np.random.uniform(-max_shift_y, max_shift_y)] for pt in startpoints]
        pil_img = TF.perspective(pil_img, startpoints, endpoints)
        # Convert back to tensor.
        img = TF.to_tensor(pil_img)
    
    return img

class EventSelfSupervisedDataset(Dataset):
    """
    Dataset for self-supervised learning using event data stored in .npy files.
    
    Each file undergoes:
      1. Loading and conversion to a tensor.
      2. Intensity centering and scaling.
      3. Binarization based on polarity.
      4. Noise injection for non-activated pixels.
      5. Generation of two independently augmented views.
    """
    def __init__(self, event_dir: str, transform_event: Optional[Callable] = None, 
                 augment: bool = True, print_debug: bool = False):
        self.event_dir = event_dir
        self.event_files: List[str] = sorted(os.listdir(event_dir))
        self.transform_event = transform_event
        self.augment = augment
        self.print_debug = print_debug

    def __len__(self) -> int:
        return len(self.event_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        event_path = os.path.join(self.event_dir, self.event_files[idx])
        event_arr = np.load(event_path)
        
        if event_arr.ndim != 2:
            raise ValueError(f"Unexpected event array shape: {event_arr.shape}. Expected a 2D array.")
        
        # Convert array to tensor and center intensities.
        event_tensor = torch.from_numpy(event_arr).float()
        event_tensor = (event_tensor - 128) * 0.2
        
        # Binarize: set positive polarity to 1.
        event_tensor = (event_tensor.abs() > 0).float()
        
        # Add noise: for a small fraction of zero pixels, set value to 1.
        empty_mask = (event_tensor == 0)
        noise_mask = (torch.rand_like(event_tensor) < 0.0050) & empty_mask
        event_tensor[noise_mask] = 1.0
        
        if self.print_debug:
            print(f'Event tensor after processing: max={event_tensor.max()}, min={event_tensor.min()}')
        
        # Apply optional additional transforms.
        if self.transform_event:
            event_tensor = self.transform_event(event_tensor)
        
        # Add channel dimension: [C, H, W].
        event_tensor = event_tensor.unsqueeze(0)
        
        # Generate two augmented views.
        if self.augment:
            view1 = augment_event(event_tensor.clone())
            view2 = augment_event(event_tensor.clone())
        else:
            view1, view2 = event_tensor, event_tensor
        
        return view1, view2

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function to stack batch items into tensors.
    """
    view1_list, view2_list = zip(*batch)
    view1_batch = torch.stack(view1_list)
    view2_batch = torch.stack(view2_list)
    return view1_batch, view2_batch

if __name__ == "__main__":
    # Set the directory where the .npy event files are stored.
    event_dir = "/home/sas/event_ws/src/event_jackal/data/data/event"  # Adjust this path as needed.
    
    # Create the dataset and dataloader.
    dataset = EventSelfSupervisedDataset(event_dir, augment=True, print_debug=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    # Visualize a single batch.
    for i, (view1, view2) in enumerate(dataloader):
        print(f"Batch {i}:")
        print("View1 shape:", view1.shape)
        print("View2 shape:", view2.shape)
        
        # Visualize the first sample from the batch.
        sample_view1 = view1[0].squeeze().cpu().numpy()  # Remove channel dimension.
        sample_view2 = view2[0].squeeze().cpu().numpy()
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(sample_view1, cmap="gray")
        axs[0].set_title("Augmented View 1")
        axs[0].axis("off")
        
        axs[1].imshow(sample_view2, cmap="gray")
        axs[1].set_title("Augmented View 2")
        axs[1].axis("off")
        
        plt.tight_layout()
        plt.show()
        
        # Only visualize one batch.
        break
