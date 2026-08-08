import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from topologylayer.nn import TopKBarcodeLengths


def dice_loss(y_pred, y_true):
    """ Standard Dice Loss: 1 - 2|X ∩ Y| / (|X| + |Y|) """
    intersection = (y_pred * y_true).sum()
    total = y_pred.sum() + y_true.sum()
    return 1.0 - (2.0 * intersection + 1e-8) / (total + 1e-8)


def compute_topological_loss_single(Z_single, target_betti, dgminfo, max_k=10, device="cuda"):
    a = dgminfo(Z_single.cpu())
    
    # Dim 0
    L0 = (TopKBarcodeLengths(dim=0, k=max_k)(a) ** 2).sum().to(device)

    # Dim 1
    target_holes = target_betti[1] if isinstance(target_betti, (tuple, list)) else target_betti.get(1, 0)
    dim_1_sq_bars = TopKBarcodeLengths(dim=1, k=max_k)(a) ** 2
    bar_signs = torch.ones(max_k, device=device)
    bar_signs[:target_holes] = -1.0
    L1 = (dim_1_sq_bars.to(device) * bar_signs).sum()


def post_processing_framework(
    model: nn.Module,
    X_test_noise: torch.Tensor,
    X_test_clean: torch.Tensor,
    Y_test_labels: list,
    H_dict: dict,
    dgminfo_layer: nn.Module,
    num_iter_topo: int = 70,
    lr: float = 1e-5,
    lambda_topo: float = 0.02,
    max_k: int = 10,
    verbose: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    L_sqdiff_weight = 1 / lambda_topo

    l2_loss_fn = nn.MSELoss()
    mse_before_list, mse_after_list = [], []
    y_before_all, y_after_all = [], []

    if verbose:
        print(f"Running Post-Processing on {len(X_test_noise)} Samples on {device}", flush=True)

    model = model.to(device)
    model_topo = copy.deepcopy(model).to(device)
    original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for i in range(len(X_test_noise)):
        model_topo.load_state_dict(original_state)
        optimizer = torch.optim.Adam(model_topo.parameters(), lr=lr)
        sample_betti = H_dict[Y_test_labels[i]]

        model.eval()
        with torch.no_grad():
            x_in = X_test_noise[i:i+1].to(device)
            y_before = model(x_in).cpu().detach()
            original_output = y_before.clone().to(device)

        mse_before_list.append(l2_loss_fn(y_before, X_test_clean[i:i+1].cpu()).item())
        y_before_all.append(y_before)

        model_topo.train()
        for _ in range(num_iter_topo):
            optimizer.zero_grad()
            Z = model_topo(x_in)
            
            L_topo = compute_topological_loss_single(Z, sample_betti, dgminfo_layer, max_k=max_k, device=device)
            L_sqdiff = l2_loss_fn(original_output, Z) * L_sqdiff_weight
            
            (L_topo + L_sqdiff).backward()
            optimizer.step()

        model_topo.eval()
        with torch.no_grad():
            y_after = model_topo(x_in).cpu().detach()

        mse_after_list.append(l2_loss_fn(y_after, X_test_clean[i:i+1].cpu()).item())
        y_after_all.append(y_after)

        if verbose and ((i + 1) % 100 == 0 or i == 0):
            print(f"[{device}] Progress: {i + 1}/{len(X_test_noise)} samples done", flush=True)

    Y_before_tensor = torch.cat(y_before_all, dim=0)
    Y_after_tensor = torch.cat(y_after_all, dim=0)

    return mse_before_list, mse_after_list, Y_before_tensor, Y_after_tensor


def semi_supervised_framework(
    model: nn.Module,
    X_labeled_noise: torch.Tensor,
    X_labeled_clean: torch.Tensor,
    X_unlabeled_noise: torch.Tensor,
    Y_unlabeled_labels: list,
    H_dict: dict,
    dgminfo_layer: nn.Module,
    warmup_epochs: int = 100,
    semisupervised_epochs: int = 100,
    lr: float = 1e-4,
    lambda_topo: float = 0.01,
    max_k: int = 10,
    device: str = "cuda"
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Stage 1: Supervised Warmup ({warmup_epochs} Epochs)", flush=True)
    labeled_dataset = TensorDataset(X_labeled_noise, X_labeled_clean)
    labeled_loader = DataLoader(labeled_dataset, batch_size=len(X_labeled_noise), shuffle=True)

    for epoch in range(1, warmup_epochs + 1):
        model.train()
        for x_b, y_b in labeled_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(x_b)
            loss = sum(dice_loss(preds[b:b+1], y_b[b:b+1]) for b in range(preds.size(0)))
            loss.backward()
            optimizer.step()

    print(f"Stage 2: Joint Semi-Supervised Training ({semisupervised_epochs} Epochs)", flush=True)
    
    x_l = X_labeled_noise.to(device)
    y_l = X_labeled_clean.to(device)
    x_u = X_unlabeled_noise.to(device)

    for epoch in range(1, semisupervised_epochs + 1):
        model.train()
        optimizer.zero_grad()

        preds_l = model(x_l)
        loss_dice = sum(dice_loss(preds_l[b:b+1], y_l[b:b+1]) for b in range(preds_l.size(0)))

        preds_u = model(x_u)
        loss_topo = sum(
            compute_topological_loss_single(
                preds_u[b:b+1], H_dict[Y_unlabeled_labels[b]], dgminfo_layer, max_k=max_k, device=device
            )
            for b in range(preds_u.size(0))
        )

        total_loss = loss_dice + lambda_topo * loss_topo
        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{semisupervised_epochs} | Sum Dice: {loss_dice.item():.4f} | Sum Topo: {loss_topo.item():.4f} | Total: {total_loss.item():.4f}", flush=True)

    return model
