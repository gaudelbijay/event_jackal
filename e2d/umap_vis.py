#!/usr/bin/env python
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D 
from torch.utils.data import DataLoader, random_split
from sklearn.decomposition import PCA
import umap.umap_ as umap  
from model import DepthNetwork
from dataloader import EventSelfSupervisedDataset, collate_fn

# Set device for computation.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path="./outputs/best_model_checkpoint.pth"):
    """
    Loads the pre-trained DepthNetwork model from a checkpoint.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint.
        
    Returns:
        model (DepthNetwork): Loaded model in evaluation mode.
    """
    # Instantiate the model.
    model = DepthNetwork(num_in_channels=1, num_out_channels=1, form_BEV=1,
                           evs_min_cutoff=1e-3, embedding_dim=1024)
    
    # Load the checkpoint. Use weights_only if available.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "network_state_dict" in checkpoint:
        state_dict = checkpoint["network_state_dict"]
    else:
        state_dict = checkpoint

    # Filter out keys not present in the current model.
    new_state_dict = {}
    for key in model.state_dict().keys():
        if key in state_dict:
            new_state_dict[key] = state_dict[key]
        else:
            print(f"Warning: Missing key in checkpoint: {key}")

    # Load the filtered state dictionary and prepare the model.
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


# Load the model.
model = load_model()


def compute_embeddings(data):
    """
    Computes embeddings using the model's embed() function.
    
    Args:
        data (torch.Tensor): Input tensor of shape [B, 1, H, W].
        
    Returns:
        np.ndarray: Computed embeddings as a numpy array of shape [B, D].
    """
    with torch.no_grad():
        embeddings = model.embed(data.to(device))
    return embeddings.cpu().numpy()


def compute_prediction(event):
    """
    Computes the predicted depth map for a single event image.
    
    Args:
        event (torch.Tensor): Single event image tensor of shape [1, H, W] or [C, H, W].
        
    Returns:
        np.ndarray: Predicted depth map as a numpy array of shape [H, W].
    """
    # Ensure the event has a batch dimension.
    event = event.unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(event)
        if isinstance(prediction, tuple):
            prediction = prediction[0]
    # Remove batch and channel dimensions.
    prediction = prediction.squeeze(0).squeeze(0).cpu().numpy()
    return prediction


def combined_visualization(embeddings, val_dataset, selected_indices):
    """
    Generates a combined visualization consisting of:
      - A 3D scatter plot of embeddings (after dimensionality reduction via PCA and UMAP)
        where unselected points are plotted in darkslategray and selected points are highlighted with unique colors.
      - A grid showing, for each selected index, the event image and its robustly scaled predicted depth map.
        The title of each depth image is colored to indicate its corresponding embedding.
    
    Args:
        embeddings (np.ndarray): Array of embeddings of shape [num_samples, D].
        val_dataset (Dataset): The validation dataset instance.
        selected_indices (list or np.ndarray): List of indices to highlight.
    """
    # Dimensionality reduction: PCA to 50 dims, then UMAP to 3 dims.
    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(embeddings)
    reducer = umap.UMAP(n_components=3, random_state=42)
    embeddings_umap = reducer.fit_transform(embeddings_pca)
    
    # Define colors for the selected indices using the tab10 colormap.
    cmap = plt.get_cmap("tab10")
    num_selected = len(selected_indices)
    colors = [cmap(i) for i in range(num_selected)]
    
    # Create main figure with two panels.
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])
    
    # Left Panel: 3D Scatter Plot.
    ax1 = fig.add_subplot(gs[0], projection="3d")
    ax1.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], embeddings_umap[:, 2],
                c="darkslategray", s=10, alpha=0.2)
    for i, idx in enumerate(selected_indices):
        pt = embeddings_umap[idx]
        ax1.scatter(pt[0], pt[1], pt[2], c=[colors[i]], s=100, label=f"Index {idx}")
    ax1.set_title("3D UMAP Visualization", fontsize=12)
    ax1.set_xlabel("UMAP1")
    ax1.set_ylabel("UMAP2")
    ax1.set_zlabel("UMAP3")
    ax1.legend()
    
    # Right Panel: Grid of Event Images and Predicted Depths.
    inner_gs = gridspec.GridSpecFromSubplotSpec(num_selected, 2,
                                                 subplot_spec=gs[1],
                                                 wspace=0.1, hspace=0.3)
    for i, idx in enumerate(selected_indices):
        # Retrieve event sample from the validation dataset.
        event, _ = val_dataset[idx]
        if torch.is_tensor(event):
            event_img = event.squeeze(0).cpu().numpy()
        else:
            event_img = np.array(event)
        
        # Compute predicted depth.
        pred_img = compute_prediction(event)
        # Robust scaling: remove outliers using percentile clipping.
        vmin, vmax = np.percentile(pred_img, [2, 98])
        pred_clipped = np.clip(pred_img, vmin, vmax)
        pred_norm = (pred_clipped - vmin) / (vmax - vmin + 1e-8)
        
        # Compute a summary statistic (average depth) and map to a color from viridis.
        avg_depth = np.mean(pred_norm)
        depth_color = plt.get_cmap("viridis")(avg_depth)
        
        # Left column: Display the event image.
        ax_event = fig.add_subplot(inner_gs[i, 0])
        ax_event.imshow(event_img, cmap="gray")
        ax_event.set_title(f"Index {idx} Event", fontsize=8, color=colors[i])
        ax_event.axis("off")
        
        # Right column: Display the predicted depth image.
        ax_pred = fig.add_subplot(inner_gs[i, 1])
        ax_pred.imshow(pred_norm, cmap="viridis")
        ax_pred.set_title(f"Index {idx} Depth", fontsize=8, color=depth_color)
        ax_pred.axis("off")
    
    plt.suptitle("Combined Embedding and Sample Visualization", fontsize=16)
    plt.show()

if __name__ == "__main__":
    # Data configuration.
    data_config = {"event_dir": "/home/sas/backup/data/data/event"}
    train_config = {"train_split": 0.8, "batch_size": 16}
    
    # Create the dataset.
    dataset = EventSelfSupervisedDataset(data_config["event_dir"], transform_event=None, augment=True)
    train_size = int(train_config["train_split"] * len(dataset))
    _, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    
    # Create a DataLoader for the validation set.
    val_loader = DataLoader(val_dataset, batch_size=train_config["batch_size"],
                            shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    # Compute embeddings for the validation set using only the first view.
    embeddings_list = []
    for batch in val_loader:
        view1_batch, view2_batch = batch  # Use only the first view.
        emb = compute_embeddings(view1_batch)
        embeddings_list.append(emb)
    embeddings = np.concatenate(embeddings_list, axis=0)
    
    # Define representative selected indices.
    num_val = len(val_dataset)
    selected_indices = np.random.choice(num_val, size=5, replace=False)
    
    combined_visualization(embeddings, val_dataset, selected_indices)
