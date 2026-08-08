import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import v2
from topologylayer.nn import LevelSetLayer2D

from loss_functions import post_processing_framework, semi_supervised_framework

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


class MNIST_classifier(nn.Module):
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


def train_initial_model(
    model: nn.Module,
    X_train_noise: torch.Tensor,
    X_train_clean: torch.Tensor,
    batch_size: int = 50,
    num_epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 42,
    verbose: bool = True,
    device: str = "cuda"
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataset_pairs = TensorDataset(X_train_noise, X_train_clean)
    train_loader = DataLoader(
        train_dataset_pairs, 
        batch_size=batch_size, 
        shuffle=True,
        generator=g
    )

    if verbose:
        print(f"Starting Supervised Training ({num_epochs} Epochs, Device: {device})", flush=True)
    for _ in range(1, num_epochs + 1):
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

    if verbose:
        print(f"Supervised Training Complete on {device}.", flush=True)
    return model


def train_classifier(model, optimizer, X, Y, X_test, Y_test,
                      batch_size=50, num_epochs=1, verbose=False, device="cpu"):
    set_seed(random_seed)

    model.train()
    N = X.shape[0]
    num_batches = N // batch_size
    
    for e in range(num_epochs):
        if verbose and ((e + 1) % 10 == 0 or e == 0):
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

    if verbose:
        test_acc, _ = evaluate_accuracy(model, X_test, Y_test, device)
        print(f"Classifier Test Accuracy: {test_acc:.2f}%")

    return model


def evaluate_accuracy(model, X_data, Y_data, device):
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


def run_one_iteration(itr_args):
    itr, N_train, N_test, m, random_seed, device_str, classifier = itr_args
    is_first = True

    set_seed(random_seed + itr)
    device = torch.device(device_str)

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

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
        batch_size=batch_size,
        num_epochs=epochs,
        lr=learning_rate,
        seed=random_seed + itr,
        verbose=is_first,
        device=device
    )

    avg_before, avg_after, Y_before, Y_after = post_processing_framework(
        model=model,
        X_test_noise=X_test_noise,
        X_test_clean=X_test,
        Y_test_labels=Y_test,
        H_dict=H_dict,
        dgminfo_layer=dgminfo,
        verbose=is_first,
        device=device
    )

    _, correct_before = evaluate_accuracy(classifier, Y_before, Y_test, device)
    _, correct_after = evaluate_accuracy(classifier, Y_after, Y_test, device)

    return avg_before, avg_after, correct_before, correct_after
