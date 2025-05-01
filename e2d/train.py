#!/usr/bin/env python
import os
import argparse
import yaml
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.utils as vutils

from .model import DepthNetwork  
from .losses import compute_ssl_depth_loss 
from .dataloader import EventSelfSupervisedDataset, collate_fn

def save_checkpoint(checkpoint, checkpoint_path, logger):
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

def load_checkpoint(checkpoint_path, device, logger=None):
    # Use weights_only=True to avoid FutureWarning
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # If the checkpoint is a training checkpoint, extract network_state_dict
    if isinstance(ckpt, dict) and "network_state_dict" in ckpt:
        state_dict = ckpt["network_state_dict"]
    else:
        state_dict = ckpt
    new_state_dict = {}
    if torch.cuda.device_count() > 1:
        # For DataParallel models, keys should start with "module."
        for key, value in state_dict.items():
            if not key.startswith("module."):
                new_state_dict["module." + key] = value
            else:
                new_state_dict[key] = value
    else:
        # For single GPU, remove "module." prefix if present.
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value
    if logger is not None:
        logger.info(f"Loaded checkpoint from {checkpoint_path} with {len(new_state_dict)} keys.")
    return new_state_dict, ckpt

def get_state_dict(model):
    return model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

class Learner:
    def __init__(self, network, optimizer, train_loader, val_loader,
                 device, output_dir, writer, accumulation_steps=16,
                 lambda_recon=1.0, lambda_contrastive=1.0,
                 lambda_obstacle=1000, lambda_depth_struct=0.1,
                 lambda_depth_embedding=1.0, lambda_histogram=0.1,
                 lambda_cycle=0.0, mask_ratio=0.3):
        self.network = network
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        self.writer = writer
        self.accumulation_steps = accumulation_steps
        
        # Loss weights
        self.lambda_recon = lambda_recon
        self.lambda_contrastive = lambda_contrastive
        self.lambda_obstacle = lambda_obstacle
        self.lambda_depth_struct = lambda_depth_struct
        self.lambda_depth_embedding = lambda_depth_embedding
        self.lambda_histogram = lambda_histogram
        self.lambda_cycle = lambda_cycle
        
        self.mask_ratio = mask_ratio

        self.best_val_loss = float('inf')
        self.global_step = 0
        self.start_epoch = 1

        self.logger = logging.getLogger("Learner")
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.scaler = torch.amp.GradScaler() if self.device.type == 'cuda' else None

    def train_epoch(self, epoch):
        self.network.train()
        running_loss = 0.0
        running_recon_loss = 0.0
        running_obstacle_loss = 0.0
        running_contrastive_loss = 0.0
        running_depth_struct_loss = 0.0
        running_depth_embedding_loss = 0.0
        running_histogram_loss = 0.0
        running_cycle_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        self.optimizer.zero_grad()

        for batch_idx, (view1, view2) in enumerate(pbar):
            # Both views are augmented versions of the same input.
            view1 = view1.to(self.device, non_blocking=True)
            view2 = view2.to(self.device, non_blocking=True)
            # Apply a random mask to view1.
            mask = (torch.rand(view1.size(0), 1, view1.size(2), view1.size(3), device=view1.device) > self.mask_ratio).float()
            masked_view1 = view1 * mask

            # Forward pass with cycle consistency if enabled
            if self.lambda_cycle > 0:
                depth1, emb1, mini_depth1 = self.network(masked_view1, return_depth=True, return_cycle=True)
                depth2, emb2, mini_depth2 = self.network(view2, return_depth=True, return_cycle=True)
            else:
                depth1, emb1 = self.network(masked_view1, return_depth=True)
                depth2, emb2 = self.network(view2, return_depth=True)
                mini_depth1, mini_depth2 = None, None

            # Compute the self-supervised loss using both views.
            loss_outputs = compute_ssl_depth_loss(
                depth1, depth2, view1, view2, emb1, emb2,
                lambda_recon=self.lambda_recon,
                lambda_contrastive=self.lambda_contrastive,
                lambda_obstacle=self.lambda_obstacle,
                lambda_depth_struct=self.lambda_depth_struct,
                lambda_depth_embedding=self.lambda_depth_embedding,
                lambda_histogram=self.lambda_histogram,
                lambda_cycle=self.lambda_cycle,
                mini_depth1=mini_depth1,
                mini_depth2=mini_depth2,
                temperature=0.1
            )
            
            total_loss, recon_loss, obstacle_loss, contrastive_loss, \
            depth_struct_loss, depth_embedding_loss, histogram_loss, cycle_loss = loss_outputs
            
            # Scale loss for gradient accumulation
            total_loss = total_loss / self.accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            # Update running statistics
            running_loss += total_loss.item() * self.accumulation_steps
            running_recon_loss += recon_loss.item()
            running_obstacle_loss += obstacle_loss.item()
            running_contrastive_loss += contrastive_loss.item()
            running_depth_struct_loss += depth_struct_loss.item()
            running_depth_embedding_loss += depth_embedding_loss.item()
            running_histogram_loss += histogram_loss.item()
            running_cycle_loss += cycle_loss.item()
            
            self.global_step += 1

            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # Update progress bar with selected metrics
            pbar.set_postfix({
                "loss": f"{running_loss/(batch_idx+1):.4f}",
                "recon": f"{running_recon_loss/(batch_idx+1):.4f}",
                "depth_emb": f"{running_depth_embedding_loss/(batch_idx+1):.4f}",
                "hist": f"{running_histogram_loss/(batch_idx+1):.4f}",
                "contrastive": f"{running_contrastive_loss/(batch_idx+1): .4f}"
            })

            # Log metrics to TensorBoard
            if batch_idx % 10 == 0:
                # Average over batches seen so far
                step_avg = batch_idx + 1
                self.writer.add_scalar("Train/TotalLoss", running_loss/step_avg, self.global_step)
                self.writer.add_scalar("Train/ReconstructionLoss", running_recon_loss/step_avg, self.global_step)
                self.writer.add_scalar("Train/ObstacleLoss", running_obstacle_loss/step_avg, self.global_step)
                self.writer.add_scalar("Train/ContrastiveLoss", running_contrastive_loss/step_avg, self.global_step)
                self.writer.add_scalar("Train/DepthStructLoss", running_depth_struct_loss/step_avg, self.global_step)
                self.writer.add_scalar("Train/DepthEmbeddingLoss", running_depth_embedding_loss/step_avg, self.global_step)
                self.writer.add_scalar("Train/HistogramLoss", running_histogram_loss/step_avg, self.global_step)
                self.writer.add_scalar("Train/CycleLoss", running_cycle_loss/step_avg, self.global_step)

        # Calculate epoch averages
        num_batches = len(self.train_loader)
        avg_loss = running_loss / num_batches
        avg_recon_loss = running_recon_loss / num_batches
        avg_obstacle_loss = running_obstacle_loss / num_batches
        avg_contrast_loss = running_contrastive_loss / num_batches
        avg_depth_struct_loss = running_depth_struct_loss / num_batches
        avg_depth_emb_loss = running_depth_embedding_loss / num_batches
        avg_histogram_loss = running_histogram_loss / num_batches
        avg_cycle_loss = running_cycle_loss / num_batches
        
        # Log epoch summary
        self.logger.info(
            f"Epoch {epoch}: Train Loss = {avg_loss:.4f} "
            f"(Recon: {avg_recon_loss:.4f}, Obstacle: {avg_obstacle_loss:.4f}, "
            f"Contrastive: {avg_contrast_loss:.4f}, DepthStruct: {avg_depth_struct_loss:.4f}, "
            f"DepthEmb: {avg_depth_emb_loss:.4f}, Histogram: {avg_histogram_loss:.4f}, "
            f"Cycle: {avg_cycle_loss:.4f})"
        )
        return avg_loss

    def validate_epoch(self, epoch):
        self.network.eval()
        total_loss = 0.0
        total_depth_emb_correlation = 0.0
        
        with torch.no_grad():
            for view1, view2 in self.val_loader:
                view1 = view1.to(self.device, non_blocking=True)
                view2 = view2.to(self.device, non_blocking=True)
                
                # Forward pass with cycle consistency if enabled
                if self.lambda_cycle > 0:
                    depth1, emb1, mini_depth1 = self.network(view1, return_depth=True, return_cycle=True)
                    depth2, emb2, mini_depth2 = self.network(view2, return_depth=True, return_cycle=True)
                else:
                    depth1, emb1 = self.network(view1, return_depth=True)
                    depth2, emb2 = self.network(view2, return_depth=True)
                    mini_depth1, mini_depth2 = None, None
                
                # Compute validation loss
                loss_outputs = compute_ssl_depth_loss(
                    depth1, depth2, view1, view2, emb1, emb2,
                    lambda_recon=self.lambda_recon,
                    lambda_contrastive=self.lambda_contrastive,
                    lambda_obstacle=self.lambda_obstacle,
                    lambda_depth_struct=self.lambda_depth_struct,
                    lambda_depth_embedding=self.lambda_depth_embedding,
                    lambda_histogram=self.lambda_histogram,
                    lambda_cycle=self.lambda_cycle,
                    mini_depth1=mini_depth1,
                    mini_depth2=mini_depth2,
                    temperature=0.1
                )
                loss = loss_outputs[0]  # total loss is first element
                
                # Compute depth-embedding correlation as a metric
                depth1_flat = torch.nn.functional.normalize(depth1.view(depth1.size(0), -1), dim=1)
                emb_sim = torch.matmul(emb1, emb1.t())
                depth_sim = torch.matmul(depth1_flat, depth1_flat.t())
                
                # Calculate correlation between similarity matrices (off-diagonal elements only)
                mask = ~torch.eye(depth_sim.size(0), dtype=torch.bool, device=depth_sim.device)
                if mask.sum() > 0:  # Ensure there are off-diagonal elements
                    correlation = torch.nn.functional.cosine_similarity(
                        depth_sim[mask].unsqueeze(0),
                        emb_sim[mask].unsqueeze(0)
                    ).item()
                    total_depth_emb_correlation += correlation
                
                total_loss += loss.item()
        
        # Compute averages
        avg_val_loss = total_loss / len(self.val_loader)
        avg_correlation = total_depth_emb_correlation / len(self.val_loader)
        
        # Log validation results
        self.logger.info(f"Epoch {epoch}: Validation Loss = {avg_val_loss:.4f}")
        self.logger.info(f"Epoch {epoch}: Depth-Embedding Correlation = {avg_correlation:.4f}")
        
        self.writer.add_scalar("Validation/Loss", avg_val_loss, epoch)
        self.writer.add_scalar("Validation/DepthEmbeddingCorrelation", avg_correlation, epoch)
        
        # Generate visualizations
        self.save_visualization(epoch)
        
        return avg_val_loss

    def save_visualization(self, epoch):
        self.network.eval()
        with torch.no_grad():
            random_index = np.random.randint(len(self.val_loader.dataset))
            # Retrieve both augmented views.
            view1, view2 = self.val_loader.dataset[random_index]
            view1 = view1.unsqueeze(0).to(self.device, non_blocking=True)
            view2 = view2.unsqueeze(0).to(self.device, non_blocking=True)
            
            # Apply masking to the first augmented view using the defined mask ratio.
            mask = (torch.rand(view1.size(0), 1, view1.size(2), view1.size(3), device=view1.device) > self.mask_ratio).float()
            masked_view1 = view1 * mask
            
            # Get depth predictions and embeddings
            network = self.network.module if hasattr(self.network, 'module') else self.network
            
            if self.lambda_cycle > 0:
                depth_pred1, emb1, mini_depth1 = network(masked_view1, return_depth=True, return_cycle=True)
                depth_pred2, emb2, mini_depth2 = network(view2, return_depth=True, return_cycle=True)
            else:
                depth_pred1, emb1 = network(masked_view1, return_depth=True)
                depth_pred2, emb2 = network(view2, return_depth=True)
                mini_depth1, mini_depth2 = None, None
        
        # Create output directory
        vis_dir = os.path.join(self.output_dir, "visualizations", f"epoch_{epoch}")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Save depth predictions for both views
        vutils.save_image(depth_pred1, os.path.join(vis_dir, "pred_depth_view1.png"), normalize=True)
        vutils.save_image(depth_pred2, os.path.join(vis_dir, "pred_depth_view2.png"), normalize=True)
        
        # Save the corresponding input views
        vutils.save_image(view1, os.path.join(vis_dir, "input_view1.png"), normalize=True)
        vutils.save_image(view2, os.path.join(vis_dir, "input_view2.png"), normalize=True)
        
        # Save mini depth reconstructions if available
        if mini_depth1 is not None and mini_depth2 is not None:
            vutils.save_image(mini_depth1, os.path.join(vis_dir, "mini_depth1.png"), normalize=True)
            vutils.save_image(mini_depth2, os.path.join(vis_dir, "mini_depth2.png"), normalize=True)
        
        # Create visualization of depth encoding in embedding space
        # This requires more samples, so use a batch from the validation loader
        try:
            if len(self.val_loader) > 0:
                batch_view1, batch_view2 = next(iter(self.val_loader))
                batch_view1 = batch_view1.to(self.device)
                
                # Get depth and embeddings
                batch_depth1, batch_emb1 = network(batch_view1, return_depth=True)
                
                # Compute depth-embedding relationship visualization if batch is large enough
                if batch_view1.size(0) >= 8:
                    # Save embedding similarity matrix
                    emb_sim = torch.matmul(batch_emb1, batch_emb1.t())
                    emb_sim_np = emb_sim.cpu().numpy()
                    
                    # Save depth similarity matrix
                    depth_flat = torch.nn.functional.normalize(batch_depth1.view(batch_depth1.size(0), -1), dim=1)
                    depth_sim = torch.matmul(depth_flat, depth_flat.t())
                    depth_sim_np = depth_sim.cpu().numpy()
                    
                    # Create a figure with two subplots
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Plot embedding similarity matrix
                    im1 = ax1.imshow(emb_sim_np, cmap='viridis')
                    ax1.set_title('Embedding Similarity Matrix')
                    plt.colorbar(im1, ax=ax1)
                    
                    # Plot depth similarity matrix
                    im2 = ax2.imshow(depth_sim_np, cmap='viridis')
                    ax2.set_title('Depth Similarity Matrix')
                    plt.colorbar(im2, ax=ax2)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, "similarity_matrices.png"))
                    plt.close()
                   
        except:
            self.logger.warning("Could not create additional visualizations")
        
        self.logger.info(f"Saved visualization for epoch {epoch} at {vis_dir}")
        self.network.train()

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'network_state_dict': get_state_dict(self.network),
        }
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model_checkpoint.pth")
            save_checkpoint(checkpoint, best_path, self.logger)
        else:
            checkpoint_path = os.path.join(self.output_dir, f"model_epoch_{epoch}.pth")
            save_checkpoint(checkpoint, checkpoint_path, self.logger)

    def train(self, num_epochs):
        for epoch in range(self.start_epoch, num_epochs + 1):
            start_time = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)
            elapsed = time.time() - start_time
            self.logger.info(f"Epoch {epoch} completed in {elapsed:.2f} seconds.")
            if epoch % 100 == 0:
                self.save_checkpoint(epoch, is_best=True)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

def main():
    parser = argparse.ArgumentParser(
        description="Train Self-Supervised Depth Prediction with Embedding Contrastive Learning"
    )
    parser.add_argument("--config", type=str, default="./config/config.yaml",
                        help="Path to configuration YAML file")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train_config = config['train']
    data_config = config['data']
    log_config = config['logging']

    os.makedirs(log_config['output_dir'], exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(log_config['tensorboard_log_dir']))

    dataset = EventSelfSupervisedDataset(data_config['event_dir'], augment=True, print_debug=False)
    train_size = int(train_config['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True  
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True  
    )

    device = torch.device(train_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    network = DepthNetwork(num_in_channels=1, num_out_channels=1, form_BEV=1,
                           evs_min_cutoff=1e-3, embedding_dim=1024).to(device)

    if torch.cuda.device_count() > 1:
        network = torch.nn.DataParallel(network)

    optimizer = optim.Adam(network.parameters(), lr=train_config['lr'])
    accumulation_steps = train_config.get("accumulation_steps", 16)
    
    # Loss hyperparameters
    lambda_recon = train_config.get("lambda_recon", 1.0)
    lambda_contrastive = train_config.get("lambda_contrastive", 1.0)
    lambda_obstacle = train_config.get("lambda_obstacle", 100)
    lambda_depth_struct = train_config.get("lambda_depth_struct", 0.1)
    lambda_depth_embedding = train_config.get("lambda_depth_embedding", 1.0)
    lambda_histogram = train_config.get("lambda_histogram", 0.1)
    lambda_cycle = train_config.get("lambda_cycle", 0.0)
    mask_ratio = train_config.get("mask_ratio", 0.5)
    
    learner = Learner(
        network, optimizer, train_loader, val_loader, device,
        log_config['output_dir'], writer, accumulation_steps,
        lambda_recon, lambda_contrastive, lambda_obstacle, 
        lambda_depth_struct, lambda_depth_embedding,
        lambda_histogram, lambda_cycle, mask_ratio
    )

    if args.resume or train_config.get("resume", False):
        resume_path = train_config.get("checkpoint_path", None)
        if resume_path is not None and os.path.isfile(resume_path):
            ckpt_state, ckpt = load_checkpoint(resume_path, device, learner.logger)
            network.load_state_dict(ckpt_state)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            learner.start_epoch = ckpt['epoch'] + 1
            learner.global_step = ckpt['global_step']
            learner.best_val_loss = ckpt['best_val_loss']
            learner.logger.info(f"Resumed training from epoch {learner.start_epoch}")
        else:
            learner.logger.info("No checkpoint found. Starting training from scratch.")

    learner.train(train_config['epochs'])
    writer.close()

def main_wrapper():
    main()

if __name__ == "__main__":
    main_wrapper()