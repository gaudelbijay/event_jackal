import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################
# 1. Spatial Transformer (for local geometry) #
###############################################
class SpatialTransformer(nn.Module):
    """
    A simple spatial transformer that predicts an affine transform on the input.
    """
    def __init__(self, in_channels):
        super(SpatialTransformer, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, padding=3),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
        # Initialize to the identity transformation.
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)  # [B, 8, 1, 1]
        xs = xs.view(x.size(0), -1)  # [B, 8]
        theta = self.fc_loc(xs)      # [B, 6]
        theta = theta.view(-1, 2, 3)  # [B, 2, 3]
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

##################################################
# 2. Encoder: Add positional encoding & STN   #
##################################################
class DepthEncoder(nn.Module):
    """
    The encoder computes UNet bottleneck features and returns skip connections.
    It converts a single‐channel event frame into a BEV representation, adds positional encodings,
    and (optionally) applies a spatial transformer to enhance local geometric awareness.
    """
    def __init__(self, num_in_channels=1, form_BEV=1, evs_min_cutoff=1e-3, dropout_prob=0.1,
                 use_positional_encoding=True, use_spatial_transformer=True):
        super(DepthEncoder, self).__init__()
        self.num_in_channels = num_in_channels
        self.form_BEV = form_BEV
        self.evs_min_cutoff = evs_min_cutoff
        self.use_positional_encoding = use_positional_encoding
        self.use_spatial_transformer = use_spatial_transformer
        self.nonlin = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        
        # Determine effective input channels after form_BEV and (optional) positional encoding.
        if self.form_BEV == 0:
            effective_in_channels = 2
        else:
            effective_in_channels = 1
        if self.use_positional_encoding:
            effective_in_channels += 2  # add x and y coordinate channels

        # Optionally apply a spatial transformer for local geometric adaptation.
        if self.use_spatial_transformer:
            self.stn = SpatialTransformer(effective_in_channels)
        
        # Block 1
        self.conv1a = nn.Conv2d(effective_in_channels, 32, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        # Block 2
        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        # Block 3
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        # Block 4
        self.conv4a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2)
        # Bottleneck (Block 5)
        self.conv5a = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    
    def add_positional_encoding(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.size()
        device = x.device
        # Create coordinate grids in the range [-1, 1]
        y_coords = torch.linspace(-1, 1, steps=H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.linspace(-1, 1, steps=W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, x_coords, y_coords], dim=1)
    
    def form_input(self, x):
        """
        Convert a single‐channel event frame to the desired representation (BEV or otherwise)
        and then add positional encodings if enabled.
        """
        x = x.clone()
        x[x.abs() < self.evs_min_cutoff] = 0.0
        if self.form_BEV == 0:
            des_input = torch.zeros(x.size(0), 2, x.size(2), x.size(3), device=x.device)
            des_input[:, 0, :, :] = torch.where(x < 0, torch.abs(x), torch.tensor(0.0, device=x.device)).squeeze(1)
            des_input[:, 1, :, :] = torch.where(x > 0, x, torch.tensor(0.0, device=x.device)).squeeze(1)
        elif self.form_BEV == 1:
            des_input = torch.abs(x)
        elif self.form_BEV == 2:
            des_input = (x != 0).float()
        else:
            raise ValueError("form_BEV should be 0, 1, or 2")
        if self.use_positional_encoding:
            des_input = self.add_positional_encoding(des_input)
        return des_input

    def forward(self, x):
        x = self.form_input(x)
        if self.use_spatial_transformer:
            x = self.stn(x)
        # Block 1
        x1 = self.nonlin(self.conv1a(x))
        x1 = self.nonlin(self.conv1b(x1))
        x1 = self.dropout(x1)
        skip1 = x1
        x1p = self.pool1(x1)
        # Block 2
        x2 = self.nonlin(self.conv2a(x1p))
        x2 = self.nonlin(self.conv2b(x2))
        x2 = self.dropout(x2)
        skip2 = x2
        x2p = self.pool2(x2)
        # Block 3
        x3 = self.nonlin(self.conv3a(x2p))
        x3 = self.nonlin(self.conv3b(x3))
        x3 = self.dropout(x3)
        skip3 = x3
        x3p = self.pool3(x3)
        # Block 4
        x4 = self.nonlin(self.conv4a(x3p))
        x4 = self.nonlin(self.conv4b(x4))
        x4 = self.dropout(x4)
        skip4 = x4
        x4p = self.pool4(x4)
        # Bottleneck
        x5 = self.nonlin(self.conv5a(x4p))
        bottleneck = self.nonlin(self.conv5b(x5))
        return bottleneck, (skip4, skip3, skip2, skip1)

#############################################
# 3 & 5. Self-Attention with Relative Bias  #
#      and a Capsule Layer for geometry     #
#############################################
def crop_to_match(tensor, target_size):
    """
    Crops the input tensor to match the target spatial size.
    """
    _, _, h, w = tensor.size()
    target_h, target_w = target_size
    start_h = max((h - target_h) // 2, 0)
    start_w = max((w - target_w) // 2, 0)
    return tensor[:, :, start_h:start_h+target_h, start_w:start_w+target_w]

class SelfAttentionPooling(nn.Module):
    """
    Multi-head self-attention pooling with relative positional bias.
    Aggregates a spatial feature map into a single vector.
    """
    def __init__(self, in_channels, num_heads=4):
        super(SelfAttentionPooling, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.qkv_proj = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # Learnable parameter for relative positional bias (per head, for 2 coordinates)
        self.relative_bias = nn.Parameter(torch.zeros(num_heads, 2))
    
    def forward(self, x):
        """
        x: tensor of shape [B, C, H, W]
        Returns: tensor of shape [B, C, 1, 1] after pooling.
        """
        B, C, H, W = x.shape
        qkv = self.qkv_proj(x)  # [B, 3C, H, W]
        q, k, v = torch.chunk(qkv, 3, dim=1)  # Each: [B, C, H, W]

        # Flatten spatial dimensions.
        q = q.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        k = k.view(B, C, H * W).transpose(1, 2)
        v = v.view(B, C, H * W).transpose(1, 2)

        # Split into multiple heads.
        head_dim = C // self.num_heads
        q = q.view(B, H * W, self.num_heads, head_dim).transpose(1, 2)  # [B, num_heads, H*W, head_dim]
        k = k.view(B, H * W, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, H * W, self.num_heads, head_dim).transpose(1, 2)

        scale = head_dim ** -0.5
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, num_heads, H*W, H*W]

        # Compute relative positional bias.
        num_tokens = H * W
        device = x.device
        grid_y = torch.linspace(-1, 1, steps=H, device=device)
        grid_x = torch.linspace(-1, 1, steps=W, device=device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        grid = torch.stack([grid_y, grid_x], dim=-1).view(-1, 2)  # [num_tokens, 2]
        rel_coords = grid.unsqueeze(1) - grid.unsqueeze(0)  # [num_tokens, num_tokens, 2]
        # For each head, compute a bias by dotting with the learned relative_bias.
        rel_bias = (rel_coords.unsqueeze(0) * self.relative_bias.unsqueeze(1).unsqueeze(1)).sum(-1)  # [num_heads, num_tokens, num_tokens]
        attn_logits = attn_logits + rel_bias.unsqueeze(0)  # [B, num_heads, num_tokens, num_tokens]

        attn = F.softmax(attn_logits, dim=-1)
        attn_out = torch.matmul(attn, v)  # [B, num_heads, H*W, head_dim]
        attn_out = attn_out.transpose(1, 2).reshape(B, H * W, C)  # [B, H*W, C]
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        out = self.out_proj(attn_out)

        # Pool spatially
        pooled = out.view(B, C, -1).mean(dim=2, keepdim=True).unsqueeze(-1)
        return pooled

class CapsuleLayer(nn.Module):
    """
    A simple capsule layer with dynamic routing.
    """
    def __init__(self, num_caps_in, dim_caps_in, num_caps_out, dim_caps_out, num_iterations=3):
        super(CapsuleLayer, self).__init__()
        self.num_caps_in = num_caps_in
        self.dim_caps_in = dim_caps_in
        self.num_caps_out = num_caps_out
        self.dim_caps_out = dim_caps_out
        self.num_iterations = num_iterations
        # Transformation matrices.
        self.W = nn.Parameter(torch.randn(1, num_caps_in, num_caps_out, dim_caps_out, dim_caps_in))
    
    def squash(self, s, eps=1e-7):
        mag_sq = (s ** 2).sum(dim=-1, keepdim=True)
        mag = torch.sqrt(mag_sq + eps)
        v = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return v

    def forward(self, x):
        # x: [B, num_caps_in, dim_caps_in]
        B = x.size(0)
        x = x.unsqueeze(2).unsqueeze(-1)  # [B, num_caps_in, 1, dim_caps_in, 1]
        W = self.W.expand(B, -1, -1, -1, -1)  # [B, num_caps_in, num_caps_out, dim_caps_out, dim_caps_in]
        u_hat = torch.matmul(W, x).squeeze(-1)  # [B, num_caps_in, num_caps_out, dim_caps_out]
        b = torch.zeros(B, self.num_caps_in, self.num_caps_out, device=x.device)
        
        # --- Iteration 1 ---
        c1 = F.softmax(b, dim=2)              # [B, num_caps_in, num_caps_out]
        c1 = c1.unsqueeze(-1)                 # [B, num_caps_in, num_caps_out, 1]
        s1 = (c1 * u_hat).sum(dim=1)           # [B, num_caps_out, dim_caps_out]
        v1 = self.squash(s1)                   # [B, num_caps_out, dim_caps_out]
        b = b + (u_hat * v1.unsqueeze(1)).sum(dim=-1)  # Update routing logits

        # --- Iteration 2 ---
        c2 = F.softmax(b, dim=2)
        c2 = c2.unsqueeze(-1)
        s2 = (c2 * u_hat).sum(dim=1)
        v2 = self.squash(s2)
        b = b + (u_hat * v2.unsqueeze(1)).sum(dim=-1)

        # --- Iteration 3 ---
        c3 = F.softmax(b, dim=2)
        c3 = c3.unsqueeze(-1)
        s3 = (c3 * u_hat).sum(dim=1)
        v3 = self.squash(s3)
        
        return v3  # [B, num_caps_out, dim_caps_out]


class ProjectionHead(nn.Module):
    """
    JEPA projection head modified with:
      - Multi-head self-attention pooling (with relative positional bias),
      - A capsule layer to further capture geometric relationships, and
      - An MLP to produce the final embedding.
    """
    def __init__(self, in_channels=512, skip_channels=(256+128+64+32),
                 embedding_dim=1024, hidden_dim=1024, num_heads=4, num_caps=16):
        super(ProjectionHead, self).__init__()
        # Fuse bottleneck and aggregated skip features.
        self.skip_fuser = nn.Conv2d(in_channels + skip_channels, in_channels, kernel_size=1)
        self.bn_fuser = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Multi-head self-attention pooling.
        self.self_attn_pool = SelfAttentionPooling(in_channels, num_heads=num_heads)
        
        # After pooling, we have a vector of size in_channels.
        # Reshape it into capsules: we choose num_caps capsules so that
        # each capsule has dimension = in_channels // num_caps.
        # Then, our capsule layer outputs capsules that we flatten to a vector of size embedding_dim.
        dim_caps_in = in_channels // num_caps  # e.g., 512/16 = 32
        num_caps_out = num_caps
        dim_caps_out = embedding_dim // num_caps  # e.g., 1024/16 = 64
        self.capsule = CapsuleLayer(num_caps_in=num_caps, dim_caps_in=dim_caps_in,
                                    num_caps_out=num_caps_out, dim_caps_out=dim_caps_out, num_iterations=3)
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, bottleneck, skip_connections):
        # Resize skip features to match bottleneck spatial dimensions.
        skip_features = [F.adaptive_avg_pool2d(skip, bottleneck.shape[2:]) for skip in skip_connections]
        skip_agg = torch.cat(skip_features, dim=1)
        # Fuse bottleneck with aggregated skip features.
        fused = torch.cat([bottleneck, skip_agg], dim=1)
        fused = self.relu(self.bn_fuser(self.skip_fuser(fused)))  # [B, in_channels, H, W]
        
        # Self-attention pooling.
        pooled = self.self_attn_pool(fused)  # [B, in_channels, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # [B, in_channels]
        
        # Reshape pooled vector into capsules.
        B, total_dim = pooled.shape
        num_caps = self.capsule.num_caps_in  # e.g. 16
        capsules_input = pooled.view(B, num_caps, total_dim // num_caps)  # [B, num_caps, dim_caps_in]
        
        # Capsule layer output.
        capsules_output = self.capsule(capsules_input)  # [B, num_caps, dim_caps_out]
        capsules_flat = capsules_output.view(B, -1)  # [B, embedding_dim]
        
        embedding = self.mlp(capsules_flat)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

###############################################
# 4. Depth Decoder (remains largely unchanged)#
###############################################
class DepthDecoder(nn.Module):
    """
    Reconstructs the depth map from the encoder's bottleneck features and skip connections.
    """
    def __init__(self, num_out_channels=1):
        super(DepthDecoder, self).__init__()
        self.nonlin = nn.ReLU(inplace=True)
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1a = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec1b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2a = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3a = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec3b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec4a = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.dec4b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(32, num_out_channels, kernel_size=1)
    
    def forward(self, bottleneck, skip_connections, output_size):
        skip4, skip3, skip2, skip1 = skip_connections
        
        x = self.upconv1(bottleneck)
        skip4 = crop_to_match(skip4, x.shape[2:])
        x = torch.cat([x, skip4], dim=1)
        x = self.nonlin(self.dec1a(x))
        x = self.nonlin(self.dec1b(x))
        
        x = self.upconv2(x)
        skip3 = crop_to_match(skip3, x.shape[2:])
        x = torch.cat([x, skip3], dim=1)
        x = self.nonlin(self.dec2a(x))
        x = self.nonlin(self.dec2b(x))
        
        x = self.upconv3(x)
        skip2 = crop_to_match(skip2, x.shape[2:])
        x = torch.cat([x, skip2], dim=1)
        x = self.nonlin(self.dec3a(x))
        x = self.nonlin(self.dec3b(x))
        
        x = self.upconv4(x)
        skip1 = crop_to_match(skip1, x.shape[2:])
        x = torch.cat([x, skip1], dim=1)
        x = self.nonlin(self.dec4a(x))
        x = self.nonlin(self.dec4b(x))
        
        depth = F.interpolate(self.out_conv(x), size=output_size, mode='bilinear', align_corners=False)
        return depth

###############################################
# 5. Complete Depth Network                   #
###############################################
class DepthNetwork(nn.Module):
    """
    Combines the encoder, projection head (with attention and capsules),
    and the depth decoder.
    The embed() method computes a rich embedding without running the decoder.
    """
    def __init__(self, num_in_channels=2, num_out_channels=1, form_BEV=0,
                 evs_min_cutoff=1e-3, embedding_dim=1024):
        super(DepthNetwork, self).__init__()
        self.encoder = DepthEncoder(num_in_channels, form_BEV, evs_min_cutoff)
        # Here, we assume skip channels add up to 256+128+64+32 = 480.
        self.projection = ProjectionHead(in_channels=512, skip_channels=480,
                                         embedding_dim=embedding_dim, hidden_dim=1024,
                                         num_heads=4, num_caps=16)
        self.decoder = DepthDecoder(num_out_channels)
    
    def embed(self, x):
        bottleneck, skip_connections = self.encoder(x)
        embedding = self.projection(bottleneck, skip_connections)
        return embedding

    def forward(self, x, return_depth=True):
        output_size = (x.size(2), x.size(3))
        bottleneck, skip_connections = self.encoder(x)
        embedding = self.projection(bottleneck, skip_connections)
        if return_depth:
            depth = self.decoder(bottleneck, skip_connections, output_size)
            return depth, embedding
        else:
            return embedding

###############################################
# 6. Example test run                         #
###############################################
if __name__ == "__main__":
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    height, width = 480, 640
    dummy_input = torch.randn(batch_size, 1, height, width).to(device)
    
    model = DepthNetwork(num_in_channels=1, num_out_channels=1, form_BEV=1,
                         evs_min_cutoff=1e-3, embedding_dim=1024).to(device)
    
    model.eval()
    with torch.no_grad():
        depth_pred, embedding = model(dummy_input, return_depth=True)
    
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()
        with torch.no_grad():
            depth_pred, embedding = model(dummy_input, return_depth=True)
        ender.record()
        torch.cuda.synchronize()
        inference_time = starter.elapsed_time(ender)
    else:
        start_time = time.time()
        with torch.no_grad():
            depth_pred, embedding = model(dummy_input, return_depth=True)
        inference_time = (time.time() - start_time) * 1000.0

    print("Depth prediction shape:", depth_pred.shape)  # Expected: [batch, 1, height, width]
    print("Embedding shape:", embedding.shape)          # Expected: [batch, 1024]
    print("Full forward pass inference time: {:.2f} ms".format(inference_time))
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        starter.record()
        with torch.no_grad():
            embedding_only = model.embed(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        inference_time_embed = starter.elapsed_time(ender)
    else:
        start_time = time.time()
        with torch.no_grad():
            embedding_only = model.embed(dummy_input)
        inference_time_embed = (time.time() - start_time) * 1000.0

    print("Embedding (only) shape:", embedding_only.shape)  # Expected: [batch, 1024]
    print("Embedding-only inference time: {:.2f} ms".format(inference_time_embed))
