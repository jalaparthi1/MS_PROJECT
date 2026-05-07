import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import random
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Config
# =========================================================
FEATURE_DIM = 128
CHUNK_SIZE = 8
BATCH_SIZE = 1
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 42
PATIENCE = 5

LAMBDA_ALIGN = 1.0
LAMBDA_INDEP = 0.3
LAMBDA_RECON = 0.3

NUM_WORKERS = 4

TEMPLATE_PATH = "/data/users3/jalaparthi1/FBIRN/Neuromark_sMRI_3.0_modelorder-100_3x3x3.nii"

paths = {
    "COBRE": "/data/users4/umunir1/Combined_data/COBRE/Images",
    "BSNIP": "/data/users4/umunir1/Combined_data/BSNIP/Images",
    "BSNIP2": "/data/users4/umunir1/Combined_data/BSNIP2",
    "PK": "/data/users4/umunir1/Combined_data/PK/Data",
    "FBIRN": "/data/users4/umunir1/Combined_data/FBIRN/Data"
}


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================================================
# Component groups
# =========================================================
groups = {
    "SC": [1, 3, 2],
    "HP": [68],
    "AU": [31, 35, 39, 11, 47, 18, 9, 94],
    "SM": [24, 5, 95, 8],
    "VI": [27, 64, 4, 29, 85, 84, 15, 52, 34, 88, 48, 43],
    "CC": [17, 42, 13, 20, 7, 38, 55, 49, 69, 62, 14, 10, 78],
    "PA": [37, 72, 51, 100],
    "DM": [36, 25, 56, 57, 54, 41, 67, 16],
    "CB": [22, 6, 91, 19, 87, 33, 98, 32, 77, 53, 74, 59, 66]
}

COMPONENTS_66 = [c - 1 for group in groups.values() for c in group]


# =========================================================
# Dataset
# =========================================================
class MRIDataset(Dataset):
    def __init__(self, input_paths, template_nifti, clip_range=None, normalize_input=True):
        self.input_paths = input_paths
        self.normalize_input = normalize_input

        template_nii = nib.load(template_nifti)
        template_data = template_nii.get_fdata().astype(np.float32)

        print("Template original shape:", template_data.shape)

        # Convert (D,H,W,100) -> (100,D,H,W)
        if template_data.shape[-1] == 100:
            template_data = template_data.transpose(3, 0, 1, 2)

        print("Template after transpose:", template_data.shape)

        template_66 = template_data[COMPONENTS_66]
        print("Selected 66 components shape:", template_66.shape)

        # Variance normalization per component
        var_per_component = template_66.var(axis=(1, 2, 3), keepdims=True)
        template_norm = template_66 / (np.sqrt(var_per_component) + 1e-8)

        # No sign flipping
        print("No sign flipping applied. Template kept as-is.")

        if clip_range is not None:
            template_norm = np.clip(template_norm, -clip_range, clip_range)

        self.template = torch.tensor(template_norm, dtype=torch.float32)
        self.C, self.D, self.H, self.W = self.template.shape

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        try:
            img_nii = nib.load(self.input_paths[idx])
            img_data = img_nii.get_fdata().astype(np.float32)

            if self.normalize_input:
                img_data = (img_data - img_data.mean()) / (img_data.std() + 1e-8)

            # (1,1,D,H,W) for interpolate
            img_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # resize input MRI to template size
            img_resized = F.interpolate(
                img_tensor,
                size=(self.D, self.H, self.W),
                mode='trilinear',
                align_corners=False
            ).squeeze(0)   # -> (1,D,H,W)

            return img_resized

        except Exception as e:
            print(f"Error loading {self.input_paths[idx]}: {e}")
            return torch.zeros((1, self.D, self.H, self.W), dtype=torch.float32)


# =========================================================
# Encoder
# =========================================================
class Encoder3D(nn.Module):
    def __init__(self, in_channels=1, out_features=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, out_features)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# =========================================================
# Decoder
# =========================================================
class Decoder3D(nn.Module):
    def __init__(self, in_features, out_shape):
        super().__init__()
        self.out_shape = out_shape  # (1,D,H,W)

        self.fc = nn.Linear(in_features, 64 * 4 * 4 * 4)

        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),

            nn.Conv3d(8, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 64, 4, 4, 4)
        x = self.deconv(x)
        x = F.interpolate(
            x,
            size=self.out_shape[1:],
            mode='trilinear',
            align_corners=False
        )
        return x


# =========================================================
# Learnable soft template gate
# =========================================================
class LearnableTemplateGate(nn.Module):
    def __init__(self, num_components=66):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_components, 1, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_components, 1, 1, 1, 1))

    def forward(self, template_v):
        return torch.sigmoid(self.scale * template_v + self.bias)


# =========================================================
# Utilities
# =========================================================
def get_componentwise_corr(features_a, features_b):
    a_mu = features_a - features_a.mean(dim=1, keepdim=True)
    b_mu = features_b - features_b.mean(dim=1, keepdim=True)
    return F.cosine_similarity(a_mu, b_mu, dim=1)


def independence_loss(features):
    feat_norm = F.normalize(features, p=2, dim=1)
    sim = torch.matmul(feat_norm, feat_norm.T)

    eye = torch.eye(sim.size(0), device=sim.device, dtype=sim.dtype)
    off_diag = sim * (1.0 - eye)

    return (off_diag ** 2).mean()


def encode_in_chunks(encoder, x, chunk_size=8):
    outputs = []
    for start in range(0, x.size(0), chunk_size):
        end = min(start + chunk_size, x.size(0))
        outputs.append(encoder(x[start:end]))
    return torch.cat(outputs, dim=0)


def compute_losses(x_img, encoder, decoder, gate_net, template_v, template_feats, chunk_size):
    soft_template_v = gate_net(template_v)            # (66,1,D,H,W)
    masked_input = x_img * soft_template_v            # (66,1,D,H,W)

    input_feats = encode_in_chunks(
        encoder, masked_input, chunk_size=chunk_size
    )

    loss_align = 1.0 - get_componentwise_corr(input_feats, template_feats).mean()
    loss_indep = independence_loss(input_feats)

    latent = input_feats.reshape(1, -1)
    x_recon = decoder(latent)
    loss_recon = F.mse_loss(x_recon, x_img)

    total_loss = (
        LAMBDA_ALIGN * loss_align +
        LAMBDA_INDEP * loss_indep +
        LAMBDA_RECON * loss_recon
    )

    return total_loss, loss_align, loss_indep, loss_recon


def build_dataset_splits(paths_dict, seed=42):
    random.seed(seed)

    train_paths, val_paths, test_paths = [], [], []

    print("\nCollecting files from all datasets...")
    for dataset_name, dataset_path in paths_dict.items():
        files = glob.glob(os.path.join(dataset_path, "**", "*.nii*"), recursive=True)
        files = sorted(files)
        random.shuffle(files)

        n = len(files)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        train_split = files[:n_train]
        val_split = files[n_train:n_train + n_val]
        test_split = files[n_train + n_val:]

        train_paths.extend(train_split)
        val_paths.extend(val_split)
        test_paths.extend(test_split)

        print(f"{dataset_name}: total={n}, train={len(train_split)}, val={len(val_split)}, test={len(test_split)}")

    random.shuffle(train_paths)
    random.shuffle(val_paths)
    random.shuffle(test_paths)

    print(f"\nFinal Split Sizes -> Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    return train_paths, val_paths, test_paths


def train_one_epoch(
    encoder, decoder, gate_net, dataloader, optimizer, scaler, template_v, device, chunk_size
):
    encoder.train()
    decoder.train()
    gate_net.train()

    epoch_total = 0.0
    epoch_align = 0.0
    epoch_indep = 0.0
    epoch_recon = 0.0

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            soft_template_v = gate_net(template_v)
            template_feats = encode_in_chunks(
                encoder, soft_template_v, chunk_size=chunk_size
            )

    for i, x_img in enumerate(dataloader):
        x_img = x_img.to(device, non_blocking=True)   # (1,1,D,H,W)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            loss, loss_align, loss_indep, loss_recon = compute_losses(
                x_img, encoder, decoder, gate_net, template_v, template_feats, chunk_size
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_total += loss.item()
        epoch_align += loss_align.item()
        epoch_indep += loss_indep.item()
        epoch_recon += loss_recon.item()

        if i % 10 == 0:
            print(
                f"Train Step [{i}/{len(dataloader)}] | "
                f"Total: {loss.item():.4f} | "
                f"Align: {loss_align.item():.4f} | "
                f"Indep: {loss_indep.item():.4f} | "
                f"Recon: {loss_recon.item():.4f}"
            )

        if torch.cuda.is_available() and i % 20 == 0:
            torch.cuda.empty_cache()

    n_steps = len(dataloader)
    return {
        "total": epoch_total / n_steps,
        "align": epoch_align / n_steps,
        "indep": epoch_indep / n_steps,
        "recon": epoch_recon / n_steps
    }


@torch.no_grad()
def evaluate(
    encoder, decoder, gate_net, dataloader, template_v, device, chunk_size, split_name="Val"
):
    encoder.eval()
    decoder.eval()
    gate_net.eval()

    epoch_total = 0.0
    epoch_align = 0.0
    epoch_indep = 0.0
    epoch_recon = 0.0

    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        soft_template_v = gate_net(template_v)
        template_feats = encode_in_chunks(
            encoder, soft_template_v, chunk_size=chunk_size
        )

    for x_img in dataloader:
        x_img = x_img.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            loss, loss_align, loss_indep, loss_recon = compute_losses(
                x_img, encoder, decoder, gate_net, template_v, template_feats, chunk_size
            )

        epoch_total += loss.item()
        epoch_align += loss_align.item()
        epoch_indep += loss_indep.item()
        epoch_recon += loss_recon.item()

    n_steps = len(dataloader)
    metrics = {
        "total": epoch_total / n_steps,
        "align": epoch_align / n_steps,
        "indep": epoch_indep / n_steps,
        "recon": epoch_recon / n_steps
    }

    print(
        f"{split_name} | "
        f"Total: {metrics['total']:.4f} | "
        f"Align: {metrics['align']:.4f} | "
        f"Indep: {metrics['indep']:.4f} | "
        f"Recon: {metrics['recon']:.4f}"
    )

    return metrics


# =========================================================
# Main
# =========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_paths, val_paths, test_paths = build_dataset_splits(paths, seed=SEED)

    if len(train_paths) == 0:
        raise RuntimeError("No MRI files found. Check dataset paths.")

    train_dataset = MRIDataset(
        input_paths=train_paths,
        template_nifti=TEMPLATE_PATH,
        clip_range=10.0,
        normalize_input=True
    )

    val_dataset = MRIDataset(
        input_paths=val_paths,
        template_nifti=TEMPLATE_PATH,
        clip_range=10.0,
        normalize_input=True
    )

    test_dataset = MRIDataset(
        input_paths=test_paths,
        template_nifti=TEMPLATE_PATH,
        clip_range=10.0,
        normalize_input=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    encoder = Encoder3D(in_channels=1, out_features=FEATURE_DIM).to(device)
    decoder = Decoder3D(
        in_features=66 * FEATURE_DIM,
        out_shape=(1, train_dataset.D, train_dataset.H, train_dataset.W)
    ).to(device)
    gate_net = LearnableTemplateGate(num_components=66).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(gate_net.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    template_v = train_dataset.template.unsqueeze(1).to(device)   # (66,1,D,H,W)
    assert not torch.isnan(template_v).any(), "Template contains NaNs"

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(EPOCHS):
        print(f"\n================ Epoch {epoch + 1}/{EPOCHS} ================\n")

        train_metrics = train_one_epoch(
            encoder=encoder,
            decoder=decoder,
            gate_net=gate_net,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            template_v=template_v,
            device=device,
            chunk_size=CHUNK_SIZE
        )

        print(
            f"Train | Total: {train_metrics['total']:.4f} | "
            f"Align: {train_metrics['align']:.4f} | "
            f"Indep: {train_metrics['indep']:.4f} | "
            f"Recon: {train_metrics['recon']:.4f}"
        )

        val_metrics = evaluate(
            encoder=encoder,
            decoder=decoder,
            gate_net=gate_net,
            dataloader=val_loader,
            template_v=template_v,
            device=device,
            chunk_size=CHUNK_SIZE,
            split_name="Val"
        )

        scheduler.step(val_metrics["total"])
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current LR: {current_lr:.6f}")

        history.append({
            "epoch": epoch + 1,
            "train_total": train_metrics["total"],
            "train_align": train_metrics["align"],
            "train_indep": train_metrics["indep"],
            "train_recon": train_metrics["recon"],
            "val_total": val_metrics["total"],
            "val_align": val_metrics["align"],
            "val_indep": val_metrics["indep"],
            "val_recon": val_metrics["recon"],
            "lr": current_lr
        })

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "gate_net_state_dict": gate_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history
                },
                "best_soft_encoding_model.pth"
            )
            print("Best model saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    print("\nLoading best model for final testing...")
    checkpoint = torch.load("best_soft_encoding_model.pth", map_location=device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    gate_net.load_state_dict(checkpoint["gate_net_state_dict"])

    test_metrics = evaluate(
        encoder=encoder,
        decoder=decoder,
        gate_net=gate_net,
        dataloader=test_loader,
        template_v=template_v,
        device=device,
        chunk_size=CHUNK_SIZE,
        split_name="Test"
    )

    print("\nTraining complete.")
    print(f"Best Validation Loss: {checkpoint['best_val_loss']:.4f}")
    print(
        f"Final Test | Total: {test_metrics['total']:.4f} | "
        f"Align: {test_metrics['align']:.4f} | "
        f"Indep: {test_metrics['indep']:.4f} | "
        f"Recon: {test_metrics['recon']:.4f}"
    )


if __name__ == "__main__":
    main()