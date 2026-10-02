"""
Denoising architectures, Fourier corruption, and experiment routines for MNIST (Experiment 1).
"""

import numpy as np
import random
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import v2
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


def corrupt_fourier(img, m, seed=42):
    """Simulates MRI-like artifacts by zeroing lines in the 2D Fourier domain.

    Zeros out m random rows and m random columns in the centered frequency domain,
    inverts back via 2D iFFT, and min-max normalizes to [0, 1].

    Args:
        img (np.ndarray or torch.Tensor): Grayscale 2D image (H, W).
        m (int): Number of horizontal and vertical frequency scan lines to zero.
        seed (int): Random seed for scan line selection.

    Returns:
        np.ndarray: Corrupted image of shape (H, W), values in [0, 1].
    """
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
    c_min, c_max = corrupted.min(), corrupted.max()
    
    return (corrupted - c_min) / (c_max - c_min) if c_max > c_min else corrupted


def conv_block(in_channels, out_channels, num_convs=3):
    """Constructs a block of Conv2d-BatchNorm2d-ReLU layers."""
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
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Segmenter_Unet(nn.Module):
    """U-Net architecture used for MNIST image denoising/reconstruction."""

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
        """Forward pass. Expects (B, 1, H, W), outputs continuous maps in (0, 1)."""
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


class MNIST_classifier(nn.Module):
    """Standard CNN classifier for downstream digit classification evaluation."""

    def __init__(self, img_dim, num_filters=16, num_classes=10):
        super(MNIST_classifier, self).__init__()
        self.img_dim = img_dim
        self.num_filters = num_filters
        self.num_classes = num_classes

        self.conv1_1 = nn.Conv2d(1, self.num_filters, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(self.num_filters, self.num_filters,   3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2d(self.num_filters, self.num_filters*2, 3, stride=2, padding=1)

        self.conv2_1 = nn.Conv2d(self.num_filters*2, self.num_filters*2, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(self.num_filters*2, self.num_filters*2, 3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2d(self.num_filters*2, self.num_filters*4, 3, stride=2, padding=1)

        self.conv3_1 = nn.Conv2d(self.num_filters*4, self.num_filters*4, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(self.num_filters*4, self.num_filters*4, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(self.num_filters*4, self.num_filters*4, 3, stride=1, padding=1)

        self.low_res_img_dim = self.img_dim // 4
        self.final_conv_num_filters = self.num_filters*4
        self.fc_1 = nn.Linear(self.low_res_img_dim**2 * self.final_conv_num_filters, self.final_conv_num_filters)
        self.fc_2 = nn.Linear(self.final_conv_num_filters, self.final_conv_num_filters)
        self.fc_3 = nn.Linear(self.final_conv_num_filters, self.final_conv_num_filters)

        self.fc_final = nn.Linear(self.final_conv_num_filters, self.num_classes)

    def forward(self, x):
        """Forward pass. Expects (B, 1, 28, 28), outputs unnormalized logits (B, 10)."""
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.relu(self.conv1_3(x))
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv2_3(x))
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = x.view(-1, self.low_res_img_dim**2 * self.final_conv_num_filters)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = self.fc_final(x)

        return x


def evaluate_accuracy(model, X_data, Y_data, device):
    """Computes top-1 classification accuracy and per-sample correctness flags.

    Args:
        model (nn.Module): Digit classifier.
        X_data (torch.Tensor): Images tensor of shape (N, 1, 28, 28).
        Y_data (torch.Tensor or list): True integer labels of length N.
        device (str or torch.device): Device for computation.

    Returns:
        tuple: (accuracy_pct, is_correct_mask)
            - accuracy_pct (float): Accuracy in percent (0 to 100).
            - is_correct_mask (torch.BoolTensor): Boolean vector of shape (N,).
    """
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        if not isinstance(Y_data, torch.Tensor):
            Y_data = torch.tensor(Y_data, dtype=torch.long)
        X_dev = X_data.to(device)
        Y_dev = Y_data.to(device)
        logits = model(X_dev)
        preds = logits.argmax(dim=1)
        is_correct = (preds == Y_dev).cpu()
        acc = is_correct.float().mean().item() * 100.0
    return acc, is_correct


def train_cnn(model, X, Y, batch_size, num_epochs, lr, seed=42, device="cuda"):
    """Trains the U-Net denoising model using MSE loss.

    Args:
        model (nn.Module): Segmenter_Unet instance.
        X (torch.Tensor): Corrupted input images (N, 1, 28, 28).
        Y (torch.Tensor): Clean ground-truth images (N, 1, 28, 28).
        batch_size (int): Training batch size.
        num_epochs (int): Total training epochs.
        lr (float): Learning rate for Adam optimizer.
        seed (int): Random seed.
        device (str or torch.device): Compute device.

    Returns:
        nn.Module: Trained model.
    """
    set_seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataset_pairs = TensorDataset(X, Y)
    train_loader = DataLoader(train_dataset_pairs, batch_size=batch_size, shuffle=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = F.mse_loss(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)

        avg_loss = train_loss / len(train_dataset_pairs)

    return model


def train_classifier(model, X, Y, X_test, Y_test, batch_size, num_epochs, seed=42, device="cpu"):
    """Trains or loads a cached MNIST classifier on clean training pairs.

    If './MNIST_classifier.pt' exists, it loads the saved weights directly.
    Otherwise, trains using CrossEntropyLoss and saves weights upon completion.

    Args:
        model (nn.Module): MNIST_classifier instance.
        X (torch.Tensor): Training images (N_train, 1, 28, 28).
        Y (torch.Tensor): Training labels (N_train,).
        X_test (torch.Tensor): Test images (N_test, 1, 28, 28).
        Y_test (torch.Tensor): Test labels (N_test,).
        batch_size (int): Training batch size.
        num_epochs (int): Number of epochs.
        seed (int): Random seed.
        device (str or torch.device): Compute device.

    Returns:
        nn.Module: Evaluated classifier model.
    """
    set_seed(seed)

    if os.path.exists('./MNIST_classifier.pt'):
        print("Classifier Model Imported", flush=True)
        model = MNIST_classifier(img_dim=28).to(device)
        model.load_state_dict(torch.load('./MNIST_classifier.pt'))
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    N = X.shape[0]
    num_batches = N // batch_size
    
    for e in range(num_epochs):
        if ((e + 1) % 50 == 0 or e == 0):
            print(f"Starting epoch {e + 1}/{num_epochs}")

        train_loss = 0.
        batch_indices = np.arange(N, dtype=int)
        np.random.shuffle(batch_indices)

        for b in range(num_batches):
            this_batch_indices = batch_indices[b*batch_size:(b+1)*batch_size]
            X_batch = X[this_batch_indices].to(device)
            Y_batch = Y[this_batch_indices].to(device)

            optimizer.zero_grad()
            predict_batch = model(X_batch)
            ce_loss = nn.CrossEntropyLoss()(predict_batch, Y_batch)
            train_loss += ce_loss.item()
            ce_loss.backward()
            optimizer.step()

    test_acc, _ = evaluate_accuracy(model, X_test, Y_test, device)
    print(f"Classifier Test Accuracy: {test_acc:.2f}%")

    torch.cuda.empty_cache()
    torch.save(model.state_dict(), './MNIST_classifier.pt')

    return model


def run_one_iteration(itr_args):
    """Executes a single Experiment 1 run for a specific noise level m.

    Trains the U-Net denoiser, runs topological post-processing on test images,
    and returns MSE and accuracy metrics before and after the framework.

    Args:
        itr_args (tuple): (train_dataset, test_dataset, N_train, N_test, m,
                           lambda_topo, lr_topo, iter_topo, random_seed,
                           device_str, classifier)

    Returns:
        tuple: (mse_before, mse_after, correct_before, correct_after)
    """
    (train_dataset, test_dataset, N_train, N_test, m, lambda_topo, lr_topo, iter_topo, 
     random_seed, device_str, classifier) = itr_args

    set_seed(random_seed)
    device = torch.device(device_str)

    X_train = torch.stack([train_dataset[i][0] for i in range(N_train)]).float()
    X_train_noise = torch.stack([
        torch.tensor(corrupt_fourier(img, m=m, seed=random_seed + i), dtype=torch.float32).unsqueeze(0)
        for i, img in enumerate(X_train)
    ])

    model = Segmenter_Unet(img_dim=28)
    model = model.to(device)

    learning_rate = 1e-3
    batch_size = N_train
    epochs = 1000

    print(f"Training CNN on {device} | Noise Level m={m}", flush=True)
    model = train_cnn(
        model=model,
        X=X_train_noise,
        Y=X_train,
        batch_size=batch_size,
        num_epochs=epochs,
        lr=learning_rate,
        seed=random_seed,
        device=device
    )

    X_test = torch.stack([test_dataset[i][0] for i in range(N_test)])
    Y_test = [test_dataset[i][1] for i in range(N_test)]

    X_test_noise = torch.stack([
        torch.tensor(corrupt_fourier(img, m=m, seed=random_seed + i), dtype=torch.float32).unsqueeze(0)
        for i, img in enumerate(X_test)
    ])

    # target Betti numbers (H0, H1) per digit class (0-9)
    # the authors use the same specifications
    # note that the number two sometimes contains 1 loop
    H_dict = {
        0: (1, 1), 1: (1, 0), 2: (1, 0), 3: (1, 0), 4: (1, 0),
        5: (1, 0), 6: (1, 1), 7: (1, 0), 8: (1, 2), 9: (1, 1)
    }
    dgminfo = LevelSetLayer2D(size=(28, 28), sublevel=False, maxdim=1)

    avg_before, avg_after, Y_before, Y_after = post_processing_framework(
        model=model,
        X=X_test_noise,
        Y=X_test,
        Y_labels=Y_test,
        H_dict=H_dict,
        dgminfo_layer=dgminfo,
        lambda_topo=lambda_topo,
        num_iter_topo=iter_topo,
        lr=lr_topo,
        size=None,
        max_k=15,
        verbose=True,
        device=device
    )

    _, correct_before = evaluate_accuracy(classifier, Y_before, Y_test, device)
    _, correct_after = evaluate_accuracy(classifier, Y_after, Y_test, device)

    return avg_before, avg_after, correct_before, correct_after


def run_grid_search(itr_args):
    """Executes hyperparameter grid search (iterations x learning rates) for a noise level m.

    Args:
        itr_args (tuple): (train_dataset, test_dataset, N_train, N_val, m,
                           lambda_topo, random_seed, device_str, classifier)

    Returns:
        dict: Contains 'm', evaluated 'iterations', 'rates', and resulting 2D arrays
              'mse' and 'acc'.
    """
    (train_dataset, test_dataset, N_train, N_val, m, lambda_topo, 
     random_seed, device_str, classifier) = itr_args

    set_seed(random_seed)
    device = torch.device(device_str)

    X_train = torch.stack([train_dataset[i][0] for i in range(N_train)]).float()
    X_train_noise = torch.stack([
        torch.tensor(corrupt_fourier(img, m=m, seed=random_seed + i), dtype=torch.float32).unsqueeze(0)
        for i, img in enumerate(X_train)
    ])

    model = Segmenter_Unet(img_dim=28).to(device)
    print(f"Training CNN on {device} | Noise Level m={m}", flush=True)
    model = train_cnn(
        model=model,
        X=X_train_noise,
        Y=X_train,
        batch_size=N_train,
        num_epochs=1000,
        lr=1e-3,
        seed=random_seed,
        device=device
    )

    val_indices = range(1000, 1000 + N_val)
    X_val = torch.stack([test_dataset[i][0] for i in val_indices])
    Y_val = [test_dataset[i][1] for i in val_indices]
    X_val_noise = torch.stack([
        torch.tensor(corrupt_fourier(img, m=m, seed=random_seed + i), dtype=torch.float32).unsqueeze(0)
        for i, img in enumerate(X_val)
    ])

    H_dict = {
        0: (1, 1), 1: (1, 0), 2: (1, 0), 3: (1, 0), 4: (1, 0),
        5: (1, 0), 6: (1, 1), 7: (1, 0), 8: (1, 2), 9: (1, 1)
    }
    dgminfo = LevelSetLayer2D(size=(28, 28), sublevel=False, maxdim=1)

    iterations = [20, 50, 80]
    rates = [1e-3, 1e-4, 1e-5]

    mse_matrix = np.zeros((len(iterations), len(rates)))
    acc_matrix = np.zeros((len(iterations), len(rates)))

    for i, n_iter in enumerate(iterations):
        for j, lr_val in enumerate(rates):
            _, mse_after, _, Y_after = post_processing_framework(
                model=model,
                X=X_val_noise,
                Y=X_val,
                Y_labels=Y_val,
                H_dict=H_dict,
                dgminfo_layer=dgminfo,
                lambda_topo=lambda_topo,
                num_iter_topo=n_iter,
                lr=lr_val,
                size=None,
                max_k=15,
                verbose=False,
                device=device
            )

            acc_after, _ = evaluate_accuracy(classifier, Y_after, Y_val, device)
            mean_mse = np.mean(mse_after)

            mse_matrix[i, j] = mean_mse
            acc_matrix[i, j] = acc_after
            print(f"[m={m}] iter={n_iter:2d}, lr={lr_val:.0e} -> MSE: {mean_mse:.5f} | Acc: {acc_after:.2f}%", flush=True)

    return {"m": m, "iterations": iterations, "rates": rates, "mse": mse_matrix, "acc": acc_matrix}