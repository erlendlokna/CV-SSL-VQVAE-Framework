import torch
from torch import Tensor
from torch import nn
from torch import relu
import torch
from torch import nn
import torch.nn.functional as F

class Projector(nn.Module):
    def __init__(self, last_channels_enc, proj_hid, proj_out, device):
        super().__init__()
        self.proj_out = proj_out #number of projected features

        self.device = device  # Store the device
        # define layers
        self.linear1 = nn.Linear(last_channels_enc, proj_hid)
        self.nl1 = nn.BatchNorm1d(proj_hid)
        self.linear2 = nn.Linear(proj_hid, proj_hid)
        self.nl2 = nn.BatchNorm1d(proj_hid)
        self.linear3 = nn.Linear(proj_hid, proj_out)

    def forward(self, x):
        x = x.to(self.device)  # Move input tensor to the device
        x = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)))  # Global max pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor
        out = relu(self.nl1(self.linear1(x)))
        out = relu(self.nl2(self.linear2(out)))
        out = self.linear3(out)
        return out
    

    
class BarlowTwins(nn.Module):
    def __init__(self, projector, lambda_ = 0.005):
        super().__init__()
        self.lambda_ = lambda_
        self.projector = projector
        self.projector_features = projector.proj_out
        self.bn = nn.BatchNorm1d(self.projector.linear3.out_features, affine=False)

    @staticmethod
    def _batch_dim_wise_normalize_z(z):
        """batch dim.-wise normalization (standard-scaling style)"""
        mean = z.mean(dim=0)  # batch-wise mean
        std = z.std(dim=0)  # batch-wise std
        norm_z = (z - mean) / std  # standard-scaling; `dim=0`: batch dim.
        return norm_z
    
    @staticmethod
    def barlow_twins_cross_correlation_mat(norm_z1: Tensor, norm_z2: Tensor) -> Tensor:
        batch_size = norm_z1.shape[0]
        C = torch.mm(norm_z1.T, norm_z2) / batch_size
        return C

    def barlow_twins_loss(self, norm_z1, norm_z2):
        C = self.barlow_twins_cross_correlation_mat(norm_z1, norm_z2)
        
        # loss
        D = C.shape[0]
        identity_mat = torch.eye(D, device=norm_z1.device)  # Specify the device here
        C_diff = (identity_mat - C) ** 2
        off_diagonal_mul = (self.lambda_ * torch.abs(identity_mat - 1)) + identity_mat
        loss = (C_diff * off_diagonal_mul).sum()
        return loss

    def forward(self, z1, z2):
        
        z1_projected = self._batch_dim_wise_normalize_z(self.projector(z1))
        z2_projected = self._batch_dim_wise_normalize_z(self.projector(z2))

        loss = self.barlow_twins_loss(z1_projected, z2_projected) 

        #scaling based on dimensionality of projector:
        loss_scaled = loss / self.projector_features

        return loss_scaled
    
    