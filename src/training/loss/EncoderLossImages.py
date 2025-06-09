import torch
import torch.nn as nn
class EncoderLossImages(nn.Module):
    """
    Encoder Loss function for STAL adapted for images.
    Loss = Mutual Information (L_MI) + Sparsity Loss (L_S).
    """
    def __init__(self, in_channels, feature_channels):
        super(EncoderLossImages, self).__init__()
        print("EncoderLoss for images initialized.")
        
    def forward(self, preds, x):
        """
        Args:
            preds: Tuple containing:
                - W: Encoded spike representation (B, C_out, H, W)
                - Z1: Feature representation of Layer 1 (optional)
                - Z2: Feature representation of Layer 2 (optional)
            x: Original image input (B, C_in, H, W)
        
        Returns:
            loss: Computed loss value.
        """
        S, Z1, Z2 = preds  # Encoded spikes and optional feature maps
        
        B, T, C_out, H, W = S.shape
        B, C_in, H_in, W_in = x.shape

        # Ensure spatial dimensions match
        assert (H, W) == (H_in, W_in), "Spatial dimensions of x and W must match."

        weights = torch.arange(1, T+1, dtype=S.dtype, device=S.device)
        weights_T = weights.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        
        # Compute weighted sum of spikes over time
        S_weighted_sumspikes = torch.sum(S * weights_T, dim=1)

        # Compute Mutual Information between projected input and spike encoding
        mi = compute_mutual_information_images(x, S_weighted_sumspikes)

        mi_Z1, mi_Z2 = 0, 0
        if Z1 is not None and Z1.shape == S.shape:
            mi_Z1 = compute_mutual_information_images(x, Z1)
        if Z2 is not None and Z2.shape == S.shape:
            mi_Z2 = compute_mutual_information_images(x, Z2)

        # Compute the sparsity constraint
        area_x = torch.sum(x, dim=(2, 3))  # Sum over spatial dimensions (B, C_out)
        n_spikes = torch.sum(S, dim=(2, 3))  # Total number of spikes per channel

        # Encourage n_spikes to match area_x (scaled by some factor)
        L_S = torch.abs(n_spikes - area_x)

        # Combine losses
        MI = mi
        cnt = 1
        if Z1 is not None:
            MI += mi_Z1
            cnt += 1
        if Z2 is not None:
            MI += mi_Z2
            cnt += 2

        # Final loss combines Mutual Information loss and Sparsity constraint
        loss = -(MI / cnt) + torch.mean(L_S)

        return loss

def compute_mutual_information_images(X, Z):
    """ 
    Computes mutual information between input images (X) and encoded spikes (Z).
    """
    # print(X, Z)
    eps = torch.tensor(1e-12, dtype=torch.float32)
    
    joint_prob = torch.mean(X * Z)  # Joint probability estimate
    px = torch.mean(X)  # Marginal probability of X
    pz = torch.mean(Z)  # Marginal probability of Z

    # Ensure probabilities are non-zero
    joint_prob = torch.max(joint_prob, eps)
    px = torch.max(px, eps)
    pz = torch.max(pz, eps)
    
    # Compute Mutual Information
    mutual_information = joint_prob * torch.log2(joint_prob / (px * pz))
    return mutual_information

def compute_mutual_information(X, Z):
    """ 
    Computes the mutual information between two random variables X and Z.
    All computations are done using torch operations, to keep the gradient flow.
    """
    if X.shape[0] != Z.shape[0]:
        raise ValueError("X and W must have the same number of samples.")
    
    if X.ndim == 3:
        if X.size(2) == 1:
            X = X.squeeze(2)
    
    eps = torch.tensor(1e-12, dtype=torch.float32)
    joint_prob = torch.mean(torch.multiply(X, Z))
    px = torch.mean(X) # Marginal probability of X
    # px = px if px > 0 else eps
    assert px >= 0, f"Marginal prob. of X is smaller than 0, px: {px}"
    pz = torch.mean(Z) # Marginal probability of Z
    assert pz >= 0, f"Marginal prob. of Z is smaller than 0, pz: {pz}"
    
    if joint_prob == 0 or px == 0 or pz == 0:
        joint_prob = torch.max(joint_prob, eps)
        px = torch.max(px, eps)
        pz = torch.max(pz, eps)
        
    mutual_information = joint_prob * torch.log2((joint_prob) / (px * pz))
    return mutual_information