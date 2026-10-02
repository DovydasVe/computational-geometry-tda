"""
Topological loss computation and test-time post-processing framework.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from topologylayer.nn import TopKBarcodeLengths


def compute_topological_loss_single(Z_single, target_betti, dgminfo, max_k=15, device="cuda"):
    """Computes topological loss for a single 2D prediction map.

    Penalizes deviations from target Betti numbers (beta_0, beta_1) by maximizing
    the lengths of the target longest bars and penalizing extraneous bars.

    Args:
        Z_single (torch.Tensor): Model output tensor for a single slice (1, C, H, W).
        target_betti (tuple or dict): Target (beta_0, beta_1) numbers.
        dgminfo (LevelSetLayer2D): TopologyLayer level-set diagram layer.
        max_k (int): Number of top persistence barcode lengths to track per dimension. For replicated experiments, 
        15 is enough to capture most topological information,
        device (str or torch.device): Device to place the loss tensor on.

    Returns:
        torch.Tensor: Scalar topological loss.
    """
    a = dgminfo(Z_single.cpu())
    
    loss_topo = torch.tensor(0.0, device=device)
    
    for dim in [0, 1]:
        target_k = target_betti[dim] if isinstance(target_betti, (tuple, list)) else target_betti.get(dim, 0)
        
        # for dim 0, the essential bar already accounts for 1 component if target_k >= 1
        effective_target = max(0, target_k - 1) if dim == 0 else target_k
        
        sq_bars = (TopKBarcodeLengths(dim=dim, k=max_k)(a) ** 2).to(device)
        
        bar_signs = torch.ones(max_k, device=device)
        if effective_target > 0:
            bar_signs[:effective_target] = -1.0
            
        loss_topo = loss_topo + (sq_bars * bar_signs).sum()

    return loss_topo


def post_processing_framework(
    model, X, Y, Y_labels, H_dict, dgminfo_layer, lambda_topo, num_iter_topo, lr, size=None, 
    max_k=10, verbose=True, device="cpu"
):
    """Performs sample-wise test-time post-processing by minimizing the topological loss.

    For each test sample, the model weights are fine-tuned from their initial state
    to minimize a combination of the topological loss and an anchor L2 loss relative
    to the initial prediction.

    Args:
        model (nn.Module): Pre-trained segmentation/denoising model.
        X (torch.Tensor): Input images tensor of shape (N, C, H, W).
        Y (torch.Tensor): Ground-truth target tensor of shape (N, C, H, W).
        Y_labels (list or array): Class labels or slice indices mapping to keys in H_dict.
        H_dict (dict): Mapping from label/index to target Betti tuple (beta_0, beta_1).
        dgminfo_layer (LevelSetLayer2D): TopologyLayer level-set layer.
        lambda_topo (float): Weighting parameter for topological regularization.
        num_iter_topo (int): Number of optimization iterations per sample.
        lr (float): Learning rate for Adam optimizer.
        size (tuple, optional): Resolution to interpolate predictions to before computing topology.
        max_k (int): Number of barcode pairs tracked per homology dimension.
        verbose (bool): Whether to log progress.
        device (str or torch.device): Compute device for model and optimization.

    Returns:
        tuple: (mse_before_list, mse_after_list, y_before_tensor, y_after_tensor)
            - mse_before_list (list of float): Sample-wise MSE before post-processing.
            - mse_after_list (list of float): Sample-wise MSE after post-processing.
            - y_before_tensor (torch.Tensor): Concatenated predictions before post-processing.
            - y_after_tensor (torch.Tensor): Concatenated predictions after post-processing.
    """
    l2_loss_fn = nn.MSELoss()
    inv_lambda = 1.0 / lambda_topo
    
    model = model.to(device).eval()
    model_topo = copy.deepcopy(model).to(device)

    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    mse_before_list, mse_after_list = [], []
    y_before_list, y_after_list = [], []

    if verbose:
        print(f"Running Post-Processing on {len(X)} Samples on {device}", flush=True)
    
    for i in range(len(X)):
        x_in = X[i:i+1].to(device)
        y_true = Y[i:i+1].to(device)
        sample_betti = H_dict[Y_labels[i]]
        with torch.no_grad():
            y_orig = model(x_in)
        
        y_before_list.append(y_orig.cpu())
        mse_before_list.append(l2_loss_fn(y_orig, y_true).item())
        model_topo.load_state_dict(original_state)
        model_topo.train()
        optimizer = torch.optim.Adam(model_topo.parameters(), lr=lr)
    
        for _ in range(num_iter_topo):
            optimizer.zero_grad()
            pred = model_topo(x_in)
            
            pred_topo = F.interpolate(pred, size=size, mode="bilinear", align_corners=False) if size else pred
            
            loss_topo = compute_topological_loss_single(pred_topo, sample_betti, dgminfo_layer, max_k=max_k, device=device)
            loss_anchor = l2_loss_fn(y_orig, pred) * inv_lambda
            
            (loss_topo + loss_anchor).backward()
            optimizer.step()

        model_topo.eval()
        with torch.no_grad():
            y_after = model_topo(x_in)
            
        y_after_list.append(y_after.cpu())
        mse_after_list.append(l2_loss_fn(y_after, y_true).item())

        if verbose and ((i + 1) % 100 == 0 or i == 0):
            print(f"[{device}] Progress: {i + 1}/{len(X)} samples done", flush=True)

    return mse_before_list, mse_after_list, torch.cat(y_before_list, dim=0), torch.cat(y_after_list, dim=0)