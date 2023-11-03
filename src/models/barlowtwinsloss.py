import torch
from torch import Tensor
from torch import nn

def batch_dim_wise_normalize_z(z):
    """batch dim.-wise normalization (standard-scaling style)"""
    mean = z.mean(dim=0)  # batch-wise mean
    std = z.std(dim=0)  # batch-wise std
    norm_z = (z - mean) / std  # standard-scaling; `dim=0`: batch dim.
    return norm_z

def barlow_twins_cross_correlation_mat(norm_z1: Tensor, norm_z2: Tensor) -> Tensor:
    batch_size = norm_z1.shape[0]
    C = torch.mm(norm_z1.T, norm_z2) / batch_size
    return C

def barlow_twins_loss(norm_z1: Tensor, norm_z2: Tensor, lambda_: float):
    C = barlow_twins_cross_correlation_mat(norm_z1, norm_z2)
    
    # loss
    D = C.shape[0]
    identity_mat = torch.eye(D, device=C.device)  # Specify the device here
    C_diff = (identity_mat - C) ** 2
    off_diagonal_mul = (lambda_ * torch.abs(identity_mat - 1)) + identity_mat
    loss = (C_diff * off_diagonal_mul).sum()
    return loss

class BarlowTwinsLoss(nn.Module):
    def __init__(self, param=0.005):
          super().__init__()
          self.param=0.005
    
    def forward(self, z1, z2):
        total_loss = 0
        
        for i in range(min(len(z1), len(z2))):
            z1i_flat = torch.flatten(z1[i], start_dim=1)
            z2i_flat = torch.flatten(z2[i], start_dim=1)
            
            z1i_flat_norm = batch_dim_wise_normalize_z(z1i_flat)
            z2i_flat_norm = batch_dim_wise_normalize_z(z2i_flat)

            total_loss += barlow_twins_loss(z1i_flat_norm, z2i_flat_norm, self.param)

        return total_loss


