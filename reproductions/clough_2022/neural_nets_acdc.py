"""
U-Net architecture, training routines, and 3D evaluation for ACDC cardiac segmentation.
"""

import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from topologylayer.nn import LevelSetLayer2D

from loss_functions import post_processing_framework

random_seed = 42

def set_seed(seed=42):
    """Sets random seeds across Python, NumPy, and PyTorch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def conv_block_batch(in_channels, out_channels, num_convs=2):
    """Constructs a block of Conv2d-BatchNorm2d-ReLU layers."""
    layers = []
    for i in range(num_convs):
        layers.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ACDC_Segmenter_Unet(nn.Module):
    """4-level U-Net segmenter with BatchNorm for 256x256 MRI images."""

    def __init__(self, img_dim=256, num_filters=32):
        super().__init__()
        self.img_dim = img_dim
        self.num_filters = num_filters
        
        self.conv1 = conv_block_batch(1, num_filters, num_convs=2)
        self.down1 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, stride=2, padding=1)
        
        self.conv2 = conv_block_batch(num_filters * 2, num_filters * 2, num_convs=2)
        self.down2 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, stride=2, padding=1)
        
        self.conv3 = conv_block_batch(num_filters * 4, num_filters * 4, num_convs=2)
        self.down3 = nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=3, stride=2, padding=1)
        
        self.conv4 = conv_block_batch(num_filters * 8, num_filters * 8, num_convs=2)
        
        self.up3 = conv_block_batch(num_filters * 12, num_filters * 4, num_convs=2)
        self.up2 = conv_block_batch(num_filters * 6, num_filters * 2, num_convs=2)
        self.up1 = conv_block_batch(num_filters * 3, num_filters, num_convs=2)
        
        self.conv_final = nn.Conv2d(num_filters, 1, kernel_size=1)

    def forward(self, x):
        """Forward pass. Expects (B, 1, 256, 256), outputs probabilities in (0, 1)."""
        x1 = self.conv1(x)
        x = F.relu(self.down1(x1))
        x2 = self.conv2(x)
        x = F.relu(self.down2(x2))
        x3 = self.conv3(x)
        x = F.relu(self.down3(x3))
        x4 = self.conv4(x)
        x = torch.cat([F.interpolate(x4, scale_factor=2), x3], dim=1)
        x = self.up3(x)
        x = torch.cat([F.interpolate(x, scale_factor=2), x2], dim=1)
        x = self.up2(x)
        x = torch.cat([F.interpolate(x, scale_factor=2), x1], dim=1)
        x = self.up1(x)
        
        return torch.sigmoid(self.conv_final(x))


def train_initial_model(
    model: nn.Module,
    X_train_noise: torch.Tensor,
    X_train_clean: torch.Tensor,
    X_val: torch.Tensor = None,
    Y_val: torch.Tensor = None,
    val_mapping: dict = None,
    batch_size: int = 64,
    num_epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
    verbose: bool = True,
    device: str = "cpu"
):
    """Trains the baseline ACDC U-Net with early stopping based on 3D volume Dice.

    Args:
        model (nn.Module): ACDC_Segmenter_Unet model.
        X_train_noise (torch.Tensor): Training slices (N, 1, 256, 256).
        X_train_clean (torch.Tensor): Ground-truth binary masks (N, 1, 256, 256).
        X_val (torch.Tensor, optional): Validation image slices.
        Y_val (torch.Tensor, optional): Validation mask slices.
        val_mapping (dict, optional): Maps (patient_id, phase) to validation slice indices.
        batch_size (int): Training batch size.
        num_epochs (int): Maximum training epochs.
        lr (float): Initial learning rate for Adam optimizer.
        patience (int): Early stopping patience on validation 3D Dice (active after epoch 50).
        seed (int): Random seed.
        verbose (bool): Whether to log training progress.
        device (str): Device for training.

    Returns:
        nn.Module: Trained model restored to best validation checkpoint.
    """
    set_seed(seed)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    g = torch.Generator()
    g.manual_seed(seed)

    train_dataset_pairs = TensorDataset(X_train_noise, X_train_clean.float())
    train_loader = DataLoader(train_dataset_pairs, batch_size=batch_size, shuffle=True, generator=g)

    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    best_val_score = 0.0
    val_score = 0.0

    if verbose:
        print(f"Starting Training on {device} (Max Epochs: {num_epochs})", flush=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_bce_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = F.binary_cross_entropy(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_bce_loss += loss.item() * x_batch.size(0)

        avg_loss = train_bce_loss / len(train_dataset_pairs)

        # early stopping logic based on 3D patient volume Dice
        if epoch > 50 and X_val is not None and Y_val is not None and val_mapping is not None:
            model.eval()
            with torch.no_grad():
                val_preds_list = []
                for b_idx in range(0, len(X_val), batch_size):
                    x_b = X_val[b_idx : b_idx + batch_size].to(device)
                    val_preds_list.append(model(x_b).cpu())
                val_preds = torch.cat(val_preds_list, dim=0)
                val_score = evaluate_3d_dice(val_preds, Y_val, val_mapping, device=device, return_mean=True)

            if val_score > best_val_score:
                best_val_score = val_score
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

        scheduler.step()

        if verbose and (epoch == 1 or epoch % 10 == 0) and X_val is not None and Y_val is not None and val_mapping is not None:
            val_str = f" | Val 3D Dice: {val_score:.4f} (Best: {best_val_score:.4f})" if epoch > 50 else ""
            print(f"  Epoch [{epoch:03d}/{num_epochs:03d}] Loss: {avg_loss:.4f}{val_str}", flush=True)

        if epoch > 50 and patience_counter >= patience:
            if verbose:
                print(f" Early stopping triggered at Epoch {epoch}! Restoring best weights (Val 3D Dice: {best_val_score:.4f}).", flush=True)
            model.load_state_dict(best_weights)
            break

    if verbose:
        print(f"Supervised Training Complete on {device}.", flush=True)
    return model


def evaluate_3d_dice(Y_pred_tensor, Y_test, patient_volumes, device="cuda", return_mean=False):
    """Computes patient-level 3D Dice coefficients from stacked 2D slices.

    Aggregates 2D slice predictions belonging to each (patient_id, phase) volume,
    thresholds at 0.5, and evaluates volumetric 3D Dice separated into End-Diastole (ED)
    and End-Systole (ES).

    Args:
        Y_pred_tensor (torch.Tensor): Continuous predictions (N, 1, 256, 256).
        Y_test (torch.Tensor): Binary ground truth (N, 1, 256, 256).
        patient_volumes (dict): Mapping (patient_id, phase) -> list of slice indices.
        device (str or torch.device): Compute device for thresholding and intersection.
        return_mean (bool): If True, returns single float of overall mean 3D Dice.

    Returns:
        tuple or float: (mean_ed_dice, mean_es_dice) if return_mean=False,
                        else float (overall_mean_dice).
    """
    ed_dices, es_dices = [], []
    for (pid, phase), slice_indices in patient_volumes.items():
        valid_indices = [idx for idx in slice_indices if idx < len(Y_pred_tensor)]
        
        if len(valid_indices) == 0:
            continue
            
        vol_pred = (Y_pred_tensor[valid_indices].to(device) > 0.5).float()
        vol_true = Y_test[valid_indices].to(device)
        
        intersection_3d = (vol_pred * vol_true).sum()
        total_3d = vol_pred.sum() + vol_true.sum()
        
        if total_3d == 0:
            continue
            
        dice_3d = ((2.0 * intersection_3d + 1e-8) / (total_3d + 1e-8)).item()
        
        if phase == "ED":
            ed_dices.append(dice_3d)
        else:
            es_dices.append(dice_3d)
            
    ed_mean = sum(ed_dices) / len(ed_dices) if len(ed_dices) > 0 else 0.0
    es_mean = sum(es_dices) / len(es_dices) if len(es_dices) > 0 else 0.0

    if return_mean:
        all_dices = ed_dices + es_dices
        return sum(all_dices) / len(all_dices) if len(all_dices) > 0 else 0.0
        
    return ed_mean, es_mean


def run_post_processing(X, Y, lambda_topo, lr_topo, iter_topo, H_dict,
                        gpu_id, slice_indices, device, seed=42):
    """Worker function executing topological post-processing on a chunk of ACDC slices.

    Loads the saved baseline model, initializes a 256x256 LevelSetLayer2D, and
    runs test-time optimization on the designated GPU worker.

    Args:
        X (torch.Tensor): Full test images tensor (N, 1, 256, 256).
        Y (torch.Tensor): Full test masks tensor (N, 1, 256, 256).
        lambda_topo (float): Topological regularization weight.
        lr_topo (float): Optimization learning rate.
        iter_topo (int): Number of post-processing iterations per slice.
        H_dict (dict): Target Betti numbers per slice index.
        gpu_id (int or str): Assigned GPU index for this worker chunk.
        slice_indices (list of int): Subset of slice indices to process.
        device (str or torch.device): Primary device for initial weights loading.
        seed (int): Random seed.

    Returns:
        tuple: (gpu_id, y_before_cpu, y_after_cpu)
    """
    set_seed(seed)
    worker_device = f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id != "cpu" else "cpu"
    
    model = ACDC_Segmenter_Unet(img_dim=256)
    state_dict = torch.load("./acdc_unet_model.pt", map_location=device)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)

    size = (256, 256)
    dgminfo = LevelSetLayer2D(size=size, sublevel=False, maxdim=1)

    _, _, y_bef, y_aft = post_processing_framework(
        model=model,
        X=X[slice_indices],
        Y=Y[slice_indices],
        Y_labels=slice_indices,
        H_dict=H_dict,
        dgminfo_layer=dgminfo,
        lambda_topo=lambda_topo,
        num_iter_topo=iter_topo,
        lr=lr_topo,
        size=size,
        max_k=15,
        verbose=True,
        device=worker_device
    )
    return gpu_id, y_bef.cpu(), y_aft.cpu()