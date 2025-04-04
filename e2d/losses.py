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

def obstacle_loss(depth, image, k=100.0, eps=1e-6):
    """
    An obstacle-aware loss that penalizes errors in regions with high image gradients,
    with increased emphasis on near obstacles. This loss is designed to emphasize regions
    where abrupt depth changes may indicate obstacles by scaling the small depth gradients
    exponentially and weighting them more if the obstacles are near (i.e. lower depth values).

    Args:
        depth (torch.Tensor): Predicted depth map, shape [B, 1, H, W].
        image (torch.Tensor): Input image, shape [B, C, H, W].
        k (float): Exponential scaling factor for the depth gradients.
        eps (float): Small constant to prevent division by zero.
    
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
    
    # Compute average depth for regions corresponding to the gradients.
    avg_depth_x = (depth[:, :, :, :-1] + depth[:, :, :, 1:]) / 2.0
    avg_depth_y = (depth[:, :, :-1, :] + depth[:, :, 1:, :]) / 2.0
    
    # Compute near obstacle weights: higher when the average depth is small.
    # We clamp the reciprocal to ensure the weights remain in a reasonable range.
    near_weight_x = torch.clamp(1.0 / (avg_depth_x + eps), min=1.0, max=10.0)
    near_weight_y = torch.clamp(1.0 / (avg_depth_y + eps), min=1.0, max=10.0)
    
    # Scale depth gradients using an exponential function.
    scaled_grad_x = torch.exp(k * depth_grad_x) - 1
    scaled_grad_y = torch.exp(k * depth_grad_y) - 1
    
    # Combine the image gradient weights with the near-obstacle weights.
    combined_weight_x = weight_x * near_weight_x
    combined_weight_y = weight_y * near_weight_y
    
    # Compute the final obstacle-aware loss.
    loss = torch.mean(scaled_grad_x * combined_weight_x) + torch.mean(scaled_grad_y * combined_weight_y)
    return loss


def intra_view_depth_embedding_consistency(depth, emb):
    """
    Ensures embedding space captures depth structure by comparing 
    similarity relationships within each view.
    
    Args:
        depth (torch.Tensor): Depth map, shape [B, 1, H, W]
        emb (torch.Tensor): Embedding vectors, shape [B, D]
    
    Returns:
        torch.Tensor: Consistency loss between depth and embedding space
    """
    # Normalize depth maps
    depth_flat = F.normalize(depth.view(depth.size(0), -1), dim=1)
    
    # Compute depth similarities (all pairs within the batch)
    depth_sim = torch.matmul(depth_flat, depth_flat.t())
    
    # Compute embedding similarities
    emb_sim = torch.matmul(emb, emb.t())
    
    # The loss is the MSE between similarity matrices
    # Exclude the diagonal (self-similarity) which is always 1
    mask = ~torch.eye(depth_sim.size(0), dtype=torch.bool, device=depth_sim.device)
    loss = F.mse_loss(emb_sim[mask], depth_sim[mask])
    return loss

def depth_structure_loss(depth):
    """
    Encourages natural depth structures by penalizing unnatural depth discontinuities.
    Works within a single view, so suitable for rotated/translated viewpoints.
    
    Args:
        depth (torch.Tensor): Depth map, shape [B, 1, H, W]
    
    Returns:
        torch.Tensor: Structure regularity loss
    """
    # Compute depth gradients
    grad_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
    grad_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
    
    # Encourage smoothness with L1 regularization, but allow for edges
    # by downweighting the loss at large gradients (natural depth boundaries)
    weights_x = torch.exp(-grad_x * 10)  # Reduce penalty at strong depth edges
    weights_y = torch.exp(-grad_y * 10)
    
    smooth_loss = torch.mean(weights_x * grad_x) + torch.mean(weights_y * grad_y)
    return smooth_loss

def depth_histogram_consistency(depth1, depth2, bins=20):
    """
    Compare depth distribution statistics which are more invariant to viewpoint changes.
    Uses histogram comparison instead of direct pixel-wise comparison.
    
    Args:
        depth1, depth2 (torch.Tensor): Depth maps, shape [B, 1, H, W]
        bins (int): Number of histogram bins
    
    Returns:
        torch.Tensor: Histogram consistency loss
    """
    batch_size = depth1.size(0)
    device = depth1.device
    
    # Get min/max depth values across both maps (per batch element)
    def get_depth_range(depth):
        depth_flat = depth.view(batch_size, -1)
        min_vals = depth_flat.min(dim=1)[0]
        max_vals = depth_flat.max(dim=1)[0]
        return min_vals, max_vals
    
    min_d1, max_d1 = get_depth_range(depth1)
    min_d2, max_d2 = get_depth_range(depth2)
    
    # Use consistent range for both histograms
    min_depth = torch.min(min_d1, min_d2)
    max_depth = torch.max(max_d1, max_d2)
    
    # Create histograms
    def create_histograms(depth, min_val, max_val):
        depth_flat = depth.view(batch_size, -1)
        histograms = []
        
        for b in range(batch_size):
            # Skip if invalid range
            if min_val[b] >= max_val[b]:
                histograms.append(torch.ones(bins, device=device) / bins)
                continue
                
            # Create bin edges
            bin_edges = torch.linspace(min_val[b].item(), max_val[b].item(), bins+1, device=device)
            hist = torch.zeros(bins, device=device)
            
            # Fill histogram
            for i in range(bins):
                if i < bins-1:
                    mask = (depth_flat[b] >= bin_edges[i]) & (depth_flat[b] < bin_edges[i+1])
                else:
                    mask = (depth_flat[b] >= bin_edges[i]) & (depth_flat[b] <= bin_edges[i+1])
                hist[i] = mask.float().sum()
            
            # Normalize
            total = hist.sum()
            if total > 0:
                hist = hist / total
            else:
                hist = torch.ones_like(hist) / bins
            
            histograms.append(hist)
        
        return torch.stack(histograms)
    
    hist1 = create_histograms(depth1, min_depth, max_depth)
    hist2 = create_histograms(depth2, min_depth, max_depth)
    
    # Use Earth Mover's Distance (approximated by L1)
    # and Jensen-Shannon divergence for histogram comparison
    loss = F.l1_loss(hist1, hist2)
    
    return loss

def cycle_consistency_loss(depth, mini_depth):
    """
    Computes cycle consistency loss between the original depth map and
    the reconstructed depth map from embedding.
    
    Args:
        depth (torch.Tensor): Original depth map, shape [B, 1, H, W]
        mini_depth (torch.Tensor): Depth map reconstructed from embedding,
                                   shape [B, 1, h, w] (h, w < H, W)
    
    Returns:
        torch.Tensor: Cycle consistency loss
    """
    # Downscale original depth to match mini_depth size
    h, w = mini_depth.shape[2:]
    downscaled_depth = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=False)
    
    # Normalize both for comparison
    normalized_depth = normalize_batch(downscaled_depth)
    normalized_mini = normalize_batch(mini_depth)
    
    # Compute the loss using both L1 and structural similarity
    l1_loss = F.l1_loss(normalized_mini, normalized_depth)
    ssim_value = ssim(normalized_mini, normalized_depth, data_range=1.0)
    ssim_loss = 1.0 - ssim_value
    
    # Combined loss (weight SSIM more as it captures structural patterns)
    return 0.15 * l1_loss + 0.85 * ssim_loss

def compute_ssl_depth_loss(depth1, depth2, view1, view2, emb1, emb2,
                          lambda_recon=1.0, lambda_contrastive=1.0,
                          lambda_obstacle=100, lambda_depth_struct=0.1, 
                          lambda_depth_embedding=1.0, lambda_histogram=1.0,
                          lambda_cycle=0.0, lambda_consistancy=0.0,
                          mini_depth1=None, mini_depth2=None,
                          temperature=0.1):
    """
    Computes the self-supervised loss for depth and embedding contrastive learning,
    with additional losses to enforce depth information in embeddings.
    Enhanced to properly handle different viewpoints with rotations and translations.
    
    Args:
        depth1, depth2: Depth maps from two views, shape [B, 1, H, W]
        view1, view2: Input images for two views, shape [B, C, H, W]
        emb1, emb2: Embeddings from two views, shape [B, D]
        lambda_*: Various loss weights
        mini_depth1, mini_depth2: Optional reconstructed depth maps from embeddings
        temperature: Temperature for contrastive loss
        
    Returns:
        tuple: Contains total_loss and individual loss components
    """
    # Original losses - reconstruction loss between views (for self-supervision)
    recon_loss1 = multi_scale_photometric_loss(view1, view2)
    recon_loss2 = multi_scale_photometric_loss(view2, view1)
    reconstruction_loss = (recon_loss1 + recon_loss2) / 2.0
    
    # Obstacle awareness loss
    obstacle_loss1 = obstacle_loss(depth1, view1)
    obstacle_loss2 = obstacle_loss(depth2, view2)
    obstacle_loss_value = (obstacle_loss1 + obstacle_loss2) / 2.0
    
    # Contrastive learning on embeddings
    contrastive_loss_value = contrastive_loss_emb(emb1, emb2, temperature=temperature)
    
    # Intra-view depth-embedding consistency (avoids cross-view issues)
    depth_embedding_loss1 = intra_view_depth_embedding_consistency(depth1, emb1)
    depth_embedding_loss2 = intra_view_depth_embedding_consistency(depth2, emb2)
    depth_embedding_loss = (depth_embedding_loss1 + depth_embedding_loss2) / 2.0
    
    # Depth structure regularization (within each view)
    depth_struct_loss1 = depth_structure_loss(depth1)
    depth_struct_loss2 = depth_structure_loss(depth2)
    depth_struct_loss_value = (depth_struct_loss1 + depth_struct_loss2) / 2.0
    
    # Depth histogram consistency (robust to viewpoint changes)
    histogram_loss = depth_histogram_consistency(depth1, depth2)
    
    # Legacy consistency term for backward compatibility
    consistancy_loss = multi_scale_photometric_loss(depth1, depth2) if lambda_consistancy > 0 else 0.0
    
    # Optional cycle consistency loss if mini_depth is provided
    cycle_loss = 0.0
    if lambda_cycle > 0 and mini_depth1 is not None and mini_depth2 is not None:
        cycle_loss1 = cycle_consistency_loss(depth1, mini_depth1)
        cycle_loss2 = cycle_consistency_loss(depth2, mini_depth2)
        cycle_loss = (cycle_loss1 + cycle_loss2) / 2.0
    
    # Compute total loss
    total_loss = (lambda_recon * reconstruction_loss +
                  lambda_obstacle * obstacle_loss_value +
                  lambda_contrastive * contrastive_loss_value + 
                  lambda_depth_struct * depth_struct_loss_value +
                  lambda_depth_embedding * depth_embedding_loss +
                  lambda_histogram * histogram_loss +
                  lambda_consistancy * consistancy_loss +
                  lambda_cycle * cycle_loss)
    
    return (total_loss, reconstruction_loss, obstacle_loss_value, 
            contrastive_loss_value, depth_struct_loss_value, 
            depth_embedding_loss, histogram_loss, cycle_loss)

if __name__ == '__main__':
    # Dummy data for testing.
    batch_size, channels, height, width = 2, 3, 128, 128
    view1 = torch.rand(batch_size, channels, height, width)
    view2 = torch.rand(batch_size, channels, height, width)
    depth1 = torch.abs(torch.randn(batch_size, 1, height, width)) + 0.1
    depth2 = depth1 + 0.01 * torch.randn(batch_size, 1, height, width)
    emb1 = F.normalize(torch.randn(batch_size, 512), p=2, dim=1)
    emb2 = F.normalize(torch.randn(batch_size, 512), p=2, dim=1)
    
    # Generate mini depth maps for testing cycle consistency
    mini_height, mini_width = 32, 32
    mini_depth1 = F.interpolate(depth1, size=(mini_height, mini_width), mode='bilinear')
    mini_depth2 = F.interpolate(depth2, size=(mini_height, mini_width), mode='bilinear')
    
    # Test the loss function
    total_loss, recon_loss, obs_loss, cont_loss, struct_loss, depth_emb_loss, hist_loss, cycle_loss = compute_ssl_depth_loss(
        depth1, depth2, view1, view2, emb1, emb2,
        lambda_recon=1.0,
        lambda_contrastive=1.0,
        lambda_obstacle=0.5,
        lambda_depth_struct=0.1,
        lambda_depth_embedding=1.0,
        lambda_histogram=0.1,
        lambda_cycle=0.5,
        mini_depth1=mini_depth1,
        mini_depth2=mini_depth2,
        temperature=0.1
    )
    
    print("Total Loss:", total_loss.item())
    print("Reconstruction (Photometric) Loss:", recon_loss.item())
    print("Obstacle-aware Loss:", obs_loss.item())
    print("Contrastive Loss:", cont_loss.item())
    print("Depth Structure Loss:", struct_loss.item())
    print("Depth-Embedding Consistency Loss:", depth_emb_loss.item())
    print("Depth Histogram Consistency Loss:", hist_loss.item())
    print("Cycle Consistency Loss:", cycle_loss.item())
