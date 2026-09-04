import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.insert(0, os.path.dirname(__file__))
from encoder_cnn import CNNEncoder
from encoder_vit import ViTEncoder
from decoder import DualEncoderWithDecoder
from loss import DepthWeightedHuberLoss
from thermocline_head import ThermoclineHead


def build_model():
    cnn = CNNEncoder(in_channels=5, embedding_dim=512)
    vit = ViTEncoder(in_channels=5, embedding_dim=512)
    model = DualEncoderWithDecoder(cnn, vit, num_depths=15)
    return model


def make_dummy_dataset(n=256):
    """Synthetic dataset for smoke-test — replace with real GLORYS patches."""
    X = torch.randn(n, 5, 64, 64)
    Y = torch.randn(n, 15) * 10 + 15   # temps roughly 5-25 C
    return TensorDataset(X, Y)


def train(
    epochs=10,
    batch_size=16,
    lr=3e-4,
    checkpoint_dir='checkpoints',
    use_dummy=True
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Dataset
    dataset = make_dummy_dataset(256) if use_dummy else None
    n_val   = int(0.2 * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)

    # Model
    model  = build_model().to(device)
    thermo = ThermoclineHead(num_depths=15).to(device)

    # Optimizer + scheduler
    optimizer = AdamW(
        list(model.parameters()) + list(thermo.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn   = DepthWeightedHuberLoss(delta=1.0).to(device)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train(); thermo.train()
        train_losses = []
        for X, Y in train_dl:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()

            T_pred = model(X)                        # (B, 15)
            main_loss = loss_fn(T_pred, Y)

            # Auxiliary thermocline loss (soft supervision)
            td_pred, _ = thermo(T_pred)
            td_true, _ = thermo(Y)
            aux_loss   = nn.HuberLoss()(td_pred, td_true) * 0.1

            loss = main_loss + aux_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validate ---
        model.eval(); thermo.eval()
        val_losses = []
        with torch.no_grad():
            for X, Y in val_dl:
                X, Y = X.to(device), Y.to(device)
                T_pred = model(X)
                val_losses.append(loss_fn(T_pred, Y).item())

        scheduler.step()

        t_loss = np.mean(train_losses)
        v_loss = np.mean(val_losses)
        print(f'Epoch {epoch:02d}/{epochs} | train={t_loss:.4f} | val={v_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}')

        # Checkpoint best model
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            ckpt = {
                'epoch':      epoch,
                'model':      model.state_dict(),
                'thermo':     thermo.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'val_loss':   v_loss
            }
            torch.save(ckpt, os.path.join(checkpoint_dir, 'best_model.pt'))
            print(f'  -> Checkpoint saved (val={v_loss:.4f})')

    print(f'\nTraining complete. Best val loss: {best_val_loss:.4f}')
    return model, thermo


if __name__ == '__main__':
    train(epochs=10, batch_size=16, lr=3e-4, use_dummy=True)
