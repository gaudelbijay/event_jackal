#!/usr/bin/env python
import os
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # required for 3D projection in matplotlib
from sklearn.manifold import TSNE
from skimage.metrics import structural_similarity as ssim  # for computing SSIM

from model import DepthNetwork
from dataloader import EventSelfSupervisedDataset, collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path="./outputs/best_model_checkpoint.pth"):
    # Use num_in_channels=1 and form_BEV=1 to match the training configuration.
    model = DepthNetwork(num_in_channels=1, num_out_channels=1, form_BEV=1,
                           evs_min_cutoff=1e-3, embedding_dim=1024)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "student_state_dict" in checkpoint:
        state_dict = checkpoint["student_state_dict"]
    elif "teacher_state_dict" in checkpoint:
        state_dict = checkpoint["teacher_state_dict"]
    else:
        state_dict = checkpoint
    new_state_dict = {}
    for key, value in state_dict.items():
        # The checkpoint keys may be prefixed with "net.", remove it.
        if key.startswith("net."):
            new_state_dict[key[4:]] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

def compute_embeddings(data):
    with torch.no_grad():
        # data is expected to have shape [B, 1, H, W].
        embeddings = model.embed(data.to(device))
    return embeddings.cpu().numpy()

def tsne_visualization(embeddings, val_dataset, selected_indices):
    """
    Displays a 3D TSNE scatter plot (left) with colored markers for selected points,
    and on the right a panel with rows for each selected sample showing:
      - Event image (first view)
      - Predicted depth map (computed by the model)
    """
    tsne = TSNE(n_components=3, perplexity=20, learning_rate=200, n_iter=1000, random_state=42)
    embeddings_3d = tsne.fit_transform(embeddings)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Create a figure with two main panels.
    fig = plt.figure(figsize=(14, 8))
    outer_gs = gridspec.GridSpec(1, 2, width_ratios=[3, 4], wspace=0.3)
    
    # Left panel: 3D TSNE scatter plot.
    ax_tsne = fig.add_subplot(outer_gs[0], projection='3d')
    ax_tsne.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
                    s=5, alpha=0.7)
    ax_tsne.set_title("3D TSNE Visualization of Embeddings")
    ax_tsne.set_xlabel("Component 1")
    ax_tsne.set_ylabel("Component 2")
    ax_tsne.set_zlabel("Component 3")
    
    for i, idx in enumerate(selected_indices):
        color = colors[i % len(colors)]
        x, y, z = embeddings_3d[idx]
        ax_tsne.plot([x], [y], [z], 'o', markersize=8, color=color)
    
    # Right panel: Grid of images.
    # Each row shows the Event image and the Predicted Depth.
    num_rows = len(selected_indices)
    num_cols = 2  
    inner_gs = gridspec.GridSpecFromSubplotSpec(num_rows, num_cols,
                                                 subplot_spec=outer_gs[1],
                                                 wspace=0.1, hspace=0.5)
    
    for row, idx in enumerate(selected_indices):
        color = colors[row % len(colors)]
        # Retrieve the sample from the validation dataset.
        # Our dataset returns (view1, view2); we use view1 as the event image.
        event, _ = val_dataset[idx]
        if torch.is_tensor(event):
            event_img = event.squeeze(0).cpu().numpy()
        else:
            event_img = np.array(event)
        
        # Compute predicted depth.
        event_tensor = event.unsqueeze(0).to(device)  # add batch dimension
        with torch.no_grad():
            pred = model(event_tensor)
            # If the model returns a tuple, extract the depth prediction.
            if isinstance(pred, tuple):
                pred = pred[0]
        pred_img = pred.squeeze(0).squeeze(0).cpu().numpy()
        
        titles = ["Event", "Predicted Depth"]
        imgs = [event_img, pred_img]
        for col in range(num_cols):
            ax = fig.add_subplot(inner_gs[row, col])
            ax.imshow(imgs[col], cmap='gray')
            ax.axis('off')
            ax.set_title(titles[col], fontsize=10, color=color)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)
    
    plt.show()

if __name__ == "__main__":
    # Data configuration.
    data_config = {
        "event_dir": "/home/sas/backup/data/data/event"
    }
    train_config = {
        "train_split": 0.8,
        "batch_size": 16    
    }
    
    # Create the dataset (which returns two augmented views) using the event directory.
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
    
    # Randomly select 5 indices for visualization.
    num_val = len(val_dataset)
    selected_indices = np.random.choice(num_val, size=5, replace=False)
    
    tsne_visualization(embeddings, val_dataset, selected_indices)
