import os
import pickle
import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from rl_algos.base_rl_algo import ReplayBuffer
import cv2

def normalize_to_range(img, min_val=-1, max_val=1):
    """
    Normalize image to specified range
    
    Args:
        img (np.ndarray): Input image
        min_val (float): Minimum value of target range
        max_val (float): Maximum value of target range
    
    Returns:
        np.ndarray: Normalized image
    """
    img = img.astype(np.float32)
    if img.max() != img.min():  # Avoid division by zero
        img = (max_val - min_val) * (img - img.min()) / (img.max() - img.min()) + min_val
    return img

def bin_image(img, bin_size=(64, 64)):
    """
    Bin the image by averaging pixels in each bin
    
    Args:
        img (np.ndarray): Input image of shape (480, 640)
        bin_size (tuple): Target size after binning (height, width)
        
    Returns:
        np.ndarray: Binned image of shape bin_size
    """
    h, w = img.shape
    bin_h, bin_w = bin_size
    
    # Calculate size of each bin
    h_bin_size = h // bin_h
    w_bin_size = w // bin_w
    
    # Truncate image to fit exact number of bins
    h_truncated = h_bin_size * bin_h
    w_truncated = w_bin_size * bin_w
    
    # Reshape and compute mean for each bin
    img_reshaped = img[:h_truncated, :w_truncated]
    img_reshaped = img_reshaped.reshape(bin_h, h_bin_size, bin_w, w_bin_size)
    binned_img = img_reshaped.mean(axis=(1, 3))  # Use mean instead of sum for averaging
    
    return binned_img

import cv2

def preprocess_image(event_data):
    """
    Preprocess the event data:
    1. Reshape to 640x480 grayscale
    2. Normalize to [-1, 1]
    3. Bin to 64x64 using averaging
    4. Normalize binned image to [-1, 1] again
    5. Dynamically set pixel values below a threshold based on average
    6. Optionally apply erosion
    7. Flatten to 4096
    
    Args:
        event_data (np.ndarray): Raw event data of shape (307200,)
        
    Returns:
        np.ndarray: Preprocessed data of shape (4096,)
    """
    # Reshape to original image size (480x640)
    img = event_data.reshape(480, 640)
    
    # First normalization to [-1, 1]
    img_normalized = normalize_to_range(img)
    
    # Bin the normalized image to 64x64
    binned_img = bin_image(img_normalized, (64, 64))
    
    # Second normalization of binned image to [-1, 1]
    binned_normalized = normalize_to_range(binned_img)
    
    # Compute dynamic threshold based on mean and standard deviation
    mean_val = np.mean(binned_normalized)
    std_val = np.std(binned_normalized)
    
    # For example, set the threshold to mean minus one standard deviation
    threshold = mean_val - std_val
    
    # Set values below the dynamic threshold to -1
    binned_normalized[binned_normalized < threshold] = -1
    
    # Optionally apply erosion to the binary-like image using a 3x3 kernel
    binned_normalized = cv2.erode(binned_normalized, np.ones((3, 3), np.float32))
    
    # Flatten to 4096
    img_flattened = binned_normalized.flatten()
    
    return img_flattened


def plot_preprocessed_image(event_data):
    """
    Plot the preprocessed image.
    
    Args:
        event_data (np.ndarray): Flattened preprocessed image data of size 4096
    """
    # Reshape flattened image to 64x64
    img_to_plot = event_data.reshape(64, 64)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img_to_plot, cmap='gray')
    plt.title('Preprocessed Event Image (Normalized & Binned)')
    plt.axis('off')
    plt.show()

def populate_replay_buffer(data_dir, replay_buffer):
    """
    Populate replay buffer with data from pickle files in the specified directory
    
    Args:
        data_dir (str): Path to directory containing trajectory pickle files
        replay_buffer (ReplayBuffer): ReplayBuffer instance to populate
    """
    # Get all pickle files in directory
    pickle_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    
    for pkl_file in sorted(pickle_files):
        file_path = os.path.join(data_dir, pkl_file)
        print(f"Loading data from {file_path}")
        
        with open(file_path, 'rb') as f:
            traj_data = pickle.load(f)
            
        # Iterate through trajectory data
        for transition in traj_data:
            # Extract observation tuples and preprocess event data
            event_data = preprocess_image(transition['obs'][0].flatten())
            goal_action = transition['obs'][1]
            
            next_event_data = preprocess_image(transition['obs_new'][0].flatten())
            next_goal_action = transition['obs_new'][1]
            
            # Create the state tuples with preprocessed data
            state = (event_data.reshape(1, -1), goal_action)
            next_state = (next_event_data.reshape(1, -1), next_goal_action)
            
            # Extract other fields
            action = np.array(transition['action'])
            reward = transition['reward']
            done = transition['done']
            world = transition['world']
            collision_reward = transition['collision_reward']
            
            # Add to replay buffer
            replay_buffer.add(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                done=done,
                task=world,
                collision_reward=collision_reward
            )
    
    print(f"Replay buffer size after loading: {replay_buffer.size}")
    print(f"Replay buffer pointer position: {replay_buffer.ptr}")

def get_replay_buffer(state_dim, action_dim, max_size, device):
    buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=int(1e6),
        device=device
    )
    populate_replay_buffer(data_dir="./data/", replay_buffer=buffer)
    return buffer

# Example usage:
if __name__ == "__main__":
    import torch
    # Define state dimensions based on preprocessed data
    event_data_dim = (1, 4096)  # 64x64 flattened image
    goal_action_dim = (1, 4)    # Original goal action dimension
    state_dim = (event_data_dim, goal_action_dim)
    action_dim = 2
    
    # Initialize replay buffer with device handling for all platforms
    device = "cpu"
    if torch.cuda.is_available():
        if torch.cuda.get_device_capability()[0] >= 3:  # Check CUDA compute capability
            device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():  # Check for Mac
        device = "mps"
    
    print(f"Using device: {device}")
    
    buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=int(1e6),
        device=device
    )
    
    # Populate buffer with data
    data_dir = "./data/"
    populate_replay_buffer(data_dir, buffer)
    
    # After populating the buffer, if you want to plot an image from the data:
    # For demonstration, let's load one file, preprocess an image, and plot it.
    sample_file = os.path.join(data_dir, os.listdir(data_dir)[10])
    with open(sample_file, 'rb') as f:
        sample_data = pickle.load(f)
    
    # Preprocess the first observation from the sample data and plot it
    first_transition = sample_data[12]
    sample_event_data = preprocess_image(first_transition['obs'][0].flatten())
    plot_preprocessed_image(sample_event_data)
