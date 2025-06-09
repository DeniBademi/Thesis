import torch

class CE_L1Loss(torch.nn.Module):
    def __init__(self, alpha=25.0, sparsity_weight=0.1):
        super(CE_L1Loss, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.alpha = alpha
        self.sparsity_weight = sparsity_weight

    def forward(self, preds, embeddings, X, y):
        # print("preds.shape", preds.shape) # [B, n_classes]
        # print("embeddings.shape", embeddings.shape) # [T, B, N, C]
        # print("X.shape", X.shape) # [B, C, H, W]
        # print("y.shape", y.shape) # [B]
        ce_loss = self.loss_fn(preds, y)
        sparsity_loss = self.compute_sparsity_loss(embeddings, X)
        return ce_loss + self.sparsity_weight * sparsity_loss

    def compute_sparsity_loss(self, embeddings, x_input):
        # embeddings: output from STAL_SPS [T, B, N, C]
        # x_input: original input [B, C, H, W]
        
        T, B, N, C = embeddings.shape
        _, in_channels, H, W = x_input.shape

        # Calculate input area (sum across all dimensions except batch)
        area_X = torch.sum(x_input.reshape(B, -1), dim=1)
        
        # Handle embeddings shape [T, B, N, C]
        # Need to reshape to put batch first: [B, T*N*C]
        embeddings_reshaped = embeddings.permute(1, 0, 2, 3).reshape(B, -1)
        
        # Option 1: Use continuous values directly
        n_spikes_continuous = torch.sum(embeddings_reshaped, dim=1)
        
        # Option 2: Apply hard threshold
        binary_spikes = (embeddings > 0.5).float()
        binary_reshaped = binary_spikes.permute(1, 0, 2, 3).reshape(B, -1)
        n_spikes_binary = torch.sum(binary_reshaped, dim=1)
        
        # Choose based on alpha value
        n_spikes = n_spikes_binary if self.alpha > 10 else n_spikes_continuous
        
        # Calculate proper scaling factor between input pixels and embedding elements
        scale = (T * N * C) / (in_channels * H * W)
        
        # Adaptive L1 loss
        L1 = torch.abs(n_spikes - (area_X * scale))
        
        return torch.mean(L1)
    
    def compute_sparsity_loss_basic(self, embeddings):
        # embeddings: output from STAL_SPS [T, B, N, C]
        
        # Simple L1 norm - sum of absolute values
        # This directly penalizes total activation, encouraging sparsity
        
        # permute embeddings to [B, T, N, C]
        embeddings = embeddings.permute(1, 0, 2, 3)
        
        # calculate L1 loss for each sample in batch and take mean
        
        abs_embeddings = torch.abs(embeddings)
        
        L1_per_sample = torch.sum(abs_embeddings, dim=(1, 2, 3))
        
        l1_loss = torch.mean(L1_per_sample)
        
        
        return l1_loss