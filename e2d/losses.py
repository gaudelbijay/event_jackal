import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim

def contrastive_loss_emb(emb1, emb2, temperature=0.1):
    """
    Computes an NT-Xent contrastive loss between two sets of normalized embeddings.
    
    Args:
        emb1 (torch.Tensor): Embeddings from view 1, shape [B, D].
        emb2 (torch.Tensor): Embeddings from view 2, shape [B, D].
        temperature (float): Temperature scaling parameter.
    
    Returns:
        torch.Tensor: The contrastive loss.
    """
    logits = torch.matmul(emb1, emb2.t()) / temperature
    labels = torch.arange(emb1.size(0), device=emb1.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def normalize_batch(x):
    """
    Normalizes a batch of images to the [0, 1] range.
    
    Args:
        x (torch.Tensor): Images of shape [B, C, H, W].
    
    Returns:
        torch.Tensor: Normalized images in [0, 1].
    """
    B = x.size(0)
    # Compute min and max per image.
    x_min = x.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    x_max = x.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    return (x - x_min) / (x_max - x_min + 1e-8)

def multi_scale_photometric_loss(source, target, scales=[1.0, 0.5, 0.25], alpha=0.9):
    """
    Computes a multi-scale photometric loss combining SSIM and L1 losses.
    Uses TorchMetrics' SSIM function and normalizes the images to [0,1].
    
    Args:
        source (torch.Tensor): Source image, shape [B, C, H, W].
        target (torch.Tensor): Target image, shape [B, C, H, W].
        scales (list): List of scales to compute the loss.
        alpha (float): Weight for the SSIM component.
    
    Returns:
        torch.Tensor: The multi-scale photometric loss.
    """
    total_loss = 0.0
    for scale in scales:
        if scale != 1.0:
            scaled_source = F.interpolate(source, scale_factor=scale, mode='bilinear', align_corners=False)
            scaled_target = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
        else:
            scaled_source, scaled_target = source, target

        # Normalize to [0, 1]
        scaled_source = normalize_batch(scaled_source)
        scaled_target = normalize_batch(scaled_target)
        
        l1_loss = F.l1_loss(scaled_source, scaled_target)
        # TorchMetrics' ssim returns a value in [0,1]; use 1-ssim as the loss.
        ssim_value = ssim(scaled_source, scaled_target, data_range=1.0)
        ssim_loss_val = 1 - ssim_value
        total_loss += alpha * ssim_loss_val + (1 - alpha) * l1_loss
    return total_loss / len(scales)

def obstacle_loss(depth, image, k=100.0):
    """
    An obstacle-aware loss that penalizes errors in regions with high image gradients.
    This loss is designed to emphasize regions where abrupt depth changes may indicate obstacles,
    scaling the small depth gradients exponentially.

    Args:
        depth (torch.Tensor): Predicted depth map, shape [B, 1, H, W].
        image (torch.Tensor): Input image, shape [B, C, H, W].
        k (float): Exponential scaling factor for the depth gradients.
    
    Returns:
        torch.Tensor: The obstacle-aware loss.
    """
    # Compute absolute depth gradients along x and y directions.
    depth_grad_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
    depth_grad_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
    
    # Convert the input image to grayscale and compute its gradients.
    image_gray = torch.mean(image, dim=1, keepdim=True)
    image_grad_x = torch.abs(image_gray[:, :, :, :-1] - image_gray[:, :, :, 1:])
    image_grad_y = torch.abs(image_gray[:, :, :-1, :] - image_gray[:, :, 1:, :])
    
    # Use a threshold (e.g., 0.001) to define regions with significant image gradients.
    weight_x = 1 + 10 * (image_grad_x > 0.001).float()
    weight_y = 1 + 10 * (image_grad_y > 0.001).float()
    
    # Scale depth gradients using an exponential function.
    scaled_grad_x = torch.exp(k * depth_grad_x) - 1
    scaled_grad_y = torch.exp(k * depth_grad_y) - 1
    
    # Compute the final obstacle-aware loss.
    loss = torch.mean(scaled_grad_x * weight_x) + torch.mean(scaled_grad_y * weight_y)
    return loss


def compute_ssl_depth_loss(depth1, depth2, view1, view2, emb1, emb2,
                           lambda_recon=1.0, lambda_contrastive=1.0,
                           lambda_obstacle=100, lambda_consistancy=10, temperature=0.1):
    """
    Computes the self-supervised loss for depth and embedding contrastive learning.
    
    Components:
      - Multi-scale reconstruction (photometric) loss: Encourages each predicted depth to be consistent with its input view.
      - Obstacle-aware loss: Emphasizes errors near high-gradient (potential obstacle) regions.
      - Contrastive loss: NT-Xent loss on the embeddings.
    
    Args:
        depth1 (torch.Tensor): Depth from view 1, shape [B, 1, H, W].
        depth2 (torch.Tensor): Depth from view 2, shape [B, 1, H, W].
        view1 (torch.Tensor): Input image for view 1, shape [B, C, H, W].
        view2 (torch.Tensor): Input image for view 2, shape [B, C, H, W].
        emb1 (torch.Tensor): Embeddings from view 1, shape [B, D].
        emb2 (torch.Tensor): Embeddings from view 2, shape [B, D].
        lambda_recon (float): Weight for reconstruction (photometric) loss.
        lambda_contrastive (float): Weight for contrastive loss on embeddings.
        lambda_obstacle (float): Weight for the obstacle-aware loss.
        temperature (float): Temperature for contrastive loss.
        
    Returns:
        total_loss, reconstruction_loss, obstacle_loss_value, contrastive_loss_value
    """
    
    recon_loss1 = multi_scale_photometric_loss(view1, view2)
    recon_loss2 = multi_scale_photometric_loss(view2, view1)
    reconstruction_loss = (recon_loss1 + recon_loss2) / 2.0

    consistancy_loss = multi_scale_photometric_loss(depth1, depth2)
    
    obstacle_loss1 = obstacle_loss(depth1, view1)
    obstacle_loss2 = obstacle_loss(depth2, view2)
    obstacle_loss_value = (obstacle_loss1 + obstacle_loss2) / 2.0
    
    contrastive_loss_value = contrastive_loss_emb(emb1, emb2, temperature=temperature)
    
    total_loss = (lambda_recon * reconstruction_loss +
                  lambda_obstacle * obstacle_loss_value +
                  lambda_contrastive * contrastive_loss_value + 
                  lambda_consistancy*consistancy_loss)
    
    return (total_loss, reconstruction_loss, obstacle_loss_value, contrastive_loss_value, consistancy_loss)

if __name__ == '__main__':
    # Dummy data for testing.
    batch_size, channels, height, width = 2, 3, 128, 128
    view1 = torch.rand(batch_size, channels, height, width)
    view2 = torch.rand(batch_size, channels, height, width)
    depth1 = torch.abs(torch.randn(batch_size, 1, height, width)) + 0.1
    depth2 = depth1 + 0.01 * torch.randn(batch_size, 1, height, width)
    emb1 = F.normalize(torch.randn(batch_size, 512), p=2, dim=1)
    emb2 = F.normalize(torch.randn(batch_size, 512), p=2, dim=1)
    
    total_loss, recon_loss, obs_loss, c_loss = compute_ssl_depth_loss(
        depth1, depth2, view1, view2, emb1, emb2,
        lambda_recon=1.0,
        lambda_contrastive=1.0,
        lambda_obstacle=0.5,
        temperature=0.1
    )
    
    print("Total Loss:", total_loss.item())
    print("Reconstruction (Photometric) Loss:", recon_loss.item())
    print("Obstacle-aware Loss:", obs_loss.item())
    print("Contrastive Loss:", c_loss.item())
