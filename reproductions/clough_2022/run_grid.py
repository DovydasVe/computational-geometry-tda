
import sys
import os
import copy
import time
import numpy as np
from scipy import stats
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import v2

import torch.multiprocessing as mp

try:
    from topologylayer.nn import LevelSetLayer2D, TopKBarcodeLengths
except ImportError:
    sys.path.append(os.path.abspath('TopologyLayer'))
    from topologylayer.nn import LevelSetLayer2D, TopKBarcodeLengths

random_seed = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def corrupt_fourier(img, m, seed=42):
    np.random.seed(seed)
    img = np.squeeze(img)
    fft_shifted = np.fft.fftshift(np.fft.fft2(img))
    h, w = img.shape
    h_idx = np.random.choice(h, size=m, replace=False)
    v_idx = np.random.choice(w, size=m, replace=False)
    corrupted_fft = fft_shifted.copy()
    corrupted_fft[h_idx, :] = 0
    corrupted_fft[:, v_idx] = 0
    corrupted = np.abs(np.fft.ifft2(corrupted_fft))
    return corrupted / corrupted.max()


def conv_block(in_channels, out_channels, num_convs=3):
    layers = []
    for i in range(num_convs):
        layers.append(
            nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Segmenter_Unet(nn.Module):
    def __init__(self, img_dim, num_filters=16):
        super().__init__()
        self.img_dim = img_dim
        self.num_filters = num_filters
        self.conv1 = conv_block(1, num_filters, num_convs=2)
        self.down1 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(num_filters * 2, num_filters * 2, num_convs=2)
        self.down2 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, stride=2, padding=1)
        self.conv3 = conv_block(num_filters * 4, num_filters * 4, num_convs=3)
        self.up1 = conv_block(num_filters * 6, num_filters * 2, num_convs=3)
        self.up2 = conv_block(num_filters * 3, num_filters, num_convs=3)
        self.conv_final = nn.Conv2d(num_filters, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x = F.relu(self.down1(x1))
        x2 = self.conv2(x)
        x = F.relu(self.down2(x2))
        x = self.conv3(x)
        x = torch.cat([F.interpolate(x, scale_factor=2), x2], dim=1)
        x = self.up1(x)
        x = torch.cat([F.interpolate(x, scale_factor=2), x1], dim=1)
        x = self.up2(x)
        x = torch.sigmoid(self.conv_final(x))
        return x


def train_initial_model(
    model: nn.Module,
    X_train_noise: torch.Tensor,
    X_train_clean: torch.Tensor,
    X_val_noise: torch.Tensor = None,
    X_val_clean: torch.Tensor = None,
    batch_size: int = 50,
    num_epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 42,
    verbose: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_criterion = nn.MSELoss()

    train_dataset_pairs = TensorDataset(X_train_noise, X_train_clean)
    train_loader = DataLoader(
        train_dataset_pairs, 
        batch_size=batch_size, 
        shuffle=True,
        generator=g
    )

    if verbose:
        print(f"Starting Supervised Training ({num_epochs} Epochs, Device: {device})", flush=True)
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_bce_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = F.binary_cross_entropy(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_bce_loss += loss.item() * x_batch.size(0)

        avg_train_bce = train_bce_loss / len(X_train_noise)

        if verbose and (epoch % 100 == 0 or epoch == 1):
            val_info = ""
            if X_val_noise is not None and X_val_clean is not None:
                model.eval()
                with torch.no_grad():
                    val_preds = model(X_val_noise.to(device))
                    val_mse = mse_criterion(val_preds, X_val_clean.to(device)).item()
                    val_info = f"\t Validation MSE: {val_mse:.6f}"
            # print(f"Epoch [{epoch:3d}/{num_epochs:3d}] \t Train BCE: {avg_train_bce:.6f}{val_info}")

    if verbose:
        print("Supervised Training Complete.\n", flush=True)
    return model


def post_processing_single_sample(
    model: nn.Module,
    X_test_single: torch.Tensor,
    label: int,
    H_dict: dict,
    dgminfo,
    l2_loss_fn=nn.MSELoss(),
    num_iter_topo: int = 30,
    lr: float = 1e-4,
    L_sqdiff_weight: float = 50.0,
    max_k: int = 10,
    device: str = "cpu"
):
    H_i = H_dict[label]
    x_gpu = X_test_single.to(device)

    model.eval()
    with torch.no_grad():
        original_model_output = model(x_gpu).detach()

    model_topo = copy.deepcopy(model).to(device)
    model_topo.train()
    optimizer = torch.optim.Adam(model_topo.parameters(), lr=lr)

    L_list = []
    for t in range(num_iter_topo):
        optimizer.zero_grad()
        Z = model_topo(x_gpu)
        Z_cpu = Z.cpu()
        a = dgminfo(Z_cpu)

        L0 = (TopKBarcodeLengths(dim=0, k=max_k)(a)**2).sum().to(device)
        dim_1_sq_bars = TopKBarcodeLengths(dim=1, k=max_k)(a)**2
        bar_signs = torch.ones(max_k)
        bar_signs[:H_i[1]] = -1.0
        L1 = (dim_1_sq_bars * bar_signs).sum().to(device)

        L_sqdiff = l2_loss_fn(original_model_output, Z) * L_sqdiff_weight
        L = L0 + L1 + L_sqdiff
        L.backward()
        L_list.append(L.item())
        optimizer.step()

    model_topo.eval()
    with torch.no_grad():
        topo_predicted_mask = model_topo(x_gpu).cpu().detach()
        
    return topo_predicted_mask, L_list


def post_processing_framework(
    model: nn.Module,
    X_test_noise: torch.Tensor,
    X_test_clean: torch.Tensor,
    Y_test_labels: list,
    H_dict: dict,
    dgminfo_layer: nn.Module,
    num_iter_topo: int = 30,        # 100 -> 30
    lr: float = 1e-4,
    L_sqdiff_weight: float = 50.0,
    max_k: int = 10,                # 20 -> 10
    verbose: bool = True,
    device: str = "cpu"
):
    l2_loss_fn = nn.MSELoss()
    mse_before_list = []
    mse_after_list = []

    if verbose:
        print(f"Running Post-Processing on {len(X_test_noise)} Samples ({device})", flush=True)
    model = model.to(device)

    for i in range(len(X_test_noise)):
        sample_label = Y_test_labels[i]

        model.eval()
        with torch.no_grad():
            x_in = X_test_noise[i:i+1].to(device)
            y_before = model(x_in).cpu().detach()
            
        mse_before = l2_loss_fn(y_before, X_test_clean[i:i+1].cpu()).item()
        mse_before_list.append(mse_before)

        y_after, _ = post_processing_single_sample(
            model=model,
            X_test_single=X_test_noise[i:i+1],
            label=sample_label,
            H_dict=H_dict,
            dgminfo=dgminfo_layer,
            num_iter_topo=num_iter_topo,
            lr=lr,
            L_sqdiff_weight=L_sqdiff_weight,
            max_k=max_k,
            device=device
        )

        mse_after = l2_loss_fn(y_after.cpu(), X_test_clean[i:i+1].cpu()).item()
        mse_after_list.append(mse_after)

        if verbose and ((i + 1) % 100 == 0 or i == 0):
            print(f"Progress: Completed {i + 1}/{len(X_test_noise)} samples", flush=True)

    avg_before = np.mean(mse_before_list)
    avg_after = np.mean(mse_after_list)

    return mse_before_list, mse_after_list


def run_one_iteration(itr_args):
    itr, N_train, N_test, m, random_seed = itr_args
    is_first = (itr == 0)

    set_seed(random_seed + itr)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    X_test = torch.stack([test_dataset[i][0] for i in range(N_test)])
    Y_test = [test_dataset[i][1] for i in range(N_test)]

    X_test_noise = torch.stack([
        torch.tensor(corrupt_fourier(img, m=m, seed=random_seed + i), dtype=torch.float32).unsqueeze(0)
        for i, img in enumerate(X_test)
    ])

    H_dict = {
        0: (1, 1), 1: (1, 0), 2: (1, 0), 3: (1, 0), 4: (1, 0),
        5: (1, 0), 6: (1, 1), 7: (1, 0), 8: (1, 2), 9: (1, 1)
    }
    dgminfo = LevelSetLayer2D(size=(28, 28), sublevel=False, maxdim=1)

    start_idx = itr * N_train
    end_idx = start_idx + N_train

    X_train = torch.stack([train_dataset[i][0] for i in range(start_idx, end_idx)])
    X_train_noise = torch.stack([
        torch.tensor(corrupt_fourier(img, m=m, seed=random_seed + start_idx + i), dtype=torch.float32).unsqueeze(0)
        for i, img in enumerate(X_train)
    ])

    torch.manual_seed(random_seed + itr)
    model = Segmenter_Unet(img_dim=28)
    model = model.to(device)

    learning_rate = 1e-4
    batch_size = N_train
    epochs = 1000

    model = train_initial_model(
        model=model,
        X_train_noise=X_train_noise,
        X_train_clean=X_train,
        X_val_noise=X_test_noise,
        X_val_clean=X_test,
        batch_size=batch_size,
        num_epochs=epochs,
        lr=learning_rate,
        seed=random_seed + itr,
        verbose=is_first,
        device=device
    )

    avg_before, avg_after = post_processing_framework(
        model=model,
        X_test_noise=X_test_noise,
        X_test_clean=X_test,
        Y_test_labels=Y_test,
        H_dict=H_dict,
        dgminfo_layer=dgminfo,
        verbose=is_first,
        device=device
    )

    return np.mean(avg_before), np.mean(avg_after)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    N_train = 100
    N_test = 1
    # m_grid = [2, 4, 6, 8, 10, 12]
    m_grid = [2]
    iterations = 5
    num_workers = 5

    for m in m_grid:
        print(f"--- Starting {iterations}-Fold Parallel Pipeline (m={m}) ---", flush=True)
    
        tasks = [(itr, N_train, N_test, m, random_seed) for itr in range(iterations)]
        
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(run_one_iteration, tasks)

        before_list = [r[0] for r in results]
        after_list = [r[1] for r in results]

        print("\n-------- Final Results --------")

        N = iterations
        statistic, p_value = stats.wilcoxon(before_list, after_list, alternative='two-sided')
        t_critical = stats.t.ppf(0.975, df=iterations - 1)

        mean_before = np.mean(before_list)
        mean_after = np.mean(after_list)
        std_before = np.std(before_list, ddof=1)
        std_after = np.std(after_list, ddof=1)

        me_before = t_critical * std_before / np.sqrt(N)
        me_after = t_critical * std_after / np.sqrt(N)

        ci_before = (mean_before - me_before, mean_before + me_before)
        ci_after = (mean_after - me_after, mean_after + me_after)

        print(f"MSE BEFORE: {mean_before:.6f}  |  [95% CI]: ({ci_before[0]:.6f}, {ci_before[1]:.6f})")
        print(f"MSE AFTER: {mean_after:.6f}  |  [95% CI]: ({ci_after[0]:.6f}, {ci_after[1]:.6f})")
        print(f"Statistical significance: {p_value <= 0.05} (p-value {p_value})")
