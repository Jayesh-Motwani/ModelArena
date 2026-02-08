import os
import random
import math
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import timm
from sklearn.model_selection import StratifiedKFold

from facenet_pytorch import MTCNN

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception:
    A = None
    ToTensorV2 = None


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# Config
@dataclass
class CFG:
    data_root: str = "videos"
    train_real_dir: str = "videos/train/real"
    train_fake_dir: str = "videos/train/fake"
    test_dir: str = "videos/test"

    img_size: int = 224
    frames_per_video: int = 16           # sampled frames per video for training/inference
    face_margin: float = 0.25            # crop margin around face bbox
    min_face_size: int = 60              # ignore tiny detections

    backbone: str = "tf_efficientnet_b3_ns"  # good tradeoff
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 6
    num_workers: int = 0
    epochs: int = 12

    # training tricks
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    amp: bool = True

    # CV / selection
    n_folds: int = 5
    fold_to_train: int = 0

    # output
    out_dir: str = "outputs"
    model_path: str = "outputs/model_best.pt"
    test_csv_path: str = "outputs/predictions.csv"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# Video utilities
def list_videos(folder: str) -> List[str]:
    exts = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)


def safe_read_video_frames_cv2(video_path: str, num_frames: int) -> List[np.ndarray]:
    """
    Uniformly sample num_frames frames using OpenCV.
    If video shorter, it will loop / repeat.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if frame_count <= 0:
        # fallback: try reading sequentially
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if len(frames) == 0:
            return []
        idxs = np.linspace(0, len(frames) - 1, num_frames).astype(int)
        return [frames[i] for i in idxs]

    idxs = np.linspace(0, frame_count - 1, num_frames).astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        return []
    # if some frames missing, pad
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return frames[:num_frames]


# Face cropper (MTCNN)-
class FaceCropper:
    def __init__(self, device: str, img_size: int, margin: float, min_face_size: int):
        self.mtcnn = MTCNN(keep_all=True, device=device)
        self.img_size = img_size
        self.margin = margin
        self.min_face_size = min_face_size

    def crop_face(self, bgr: np.ndarray) -> np.ndarray:
        """
        Returns a face-cropped RGB image resized to img_size.
        If no face found, returns center crop of full frame.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # MTCNN expects PIL or numpy RGB; returns boxes in (x1,y1,x2,y2)
        boxes, probs = self.mtcnn.detect(rgb)
        h, w = rgb.shape[:2]

        if boxes is None or len(boxes) == 0:
            return self._fallback_center(rgb)

        # pick best box by probability
        best_i = int(np.argmax(probs))
        x1, y1, x2, y2 = boxes[best_i]

        # filter tiny faces
        bw, bh = (x2 - x1), (y2 - y1)
        if bw < self.min_face_size or bh < self.min_face_size:
            return self._fallback_center(rgb)

        # expand with margin
        mx = bw * self.margin
        my = bh * self.margin
        x1 = max(0, int(x1 - mx))
        y1 = max(0, int(y1 - my))
        x2 = min(w, int(x2 + mx))
        y2 = min(h, int(y2 + my))

        face = rgb[y1:y2, x1:x2]
        if face.size == 0:
            return self._fallback_center(rgb)

        face = cv2.resize(face, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        return face

    def _fallback_center(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        side = min(h, w)
        y1 = (h - side) // 2
        x1 = (w - side) // 2
        crop = rgb[y1:y1 + side, x1:x1 + side]
        crop = cv2.resize(crop, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        return crop


# Transforms
def build_transforms(img_size: int, train: bool):
    if A is None:
        # minimal torch conversion if albumentations not installed
        def _basic(x):
            x = x.astype(np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))
            return torch.tensor(x, dtype=torch.float32)
        return _basic

    if train:
        tfm = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.ImageCompression(quality_lower=35, quality_upper=100, p=0.35),  # deepfake-ish artifacts
            A.GaussianBlur(blur_limit=(3, 7), p=0.15),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.08, rotate_limit=8, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=24, max_width=24, p=0.2),
            A.Resize(img_size, img_size),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        tfm = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(),
            ToTensorV2(),
        ])

    def _apply(x):
        out = tfm(image=x)["image"]
        return out

    return _apply


class VideoFaceDataset(Dataset):
    def __init__(
        self,
        video_paths: List[str],
        labels: Optional[List[int]],
        face_cropper: FaceCropper,
        frames_per_video: int,
        transform,
        is_train: bool,
    ):
        self.video_paths = video_paths
        self.labels = labels
        self.face_cropper = face_cropper
        self.frames_per_video = frames_per_video
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx: int):
        vp = self.video_paths[idx]
        frames = safe_read_video_frames_cv2(vp, self.frames_per_video)
        if len(frames) == 0:
            # return blank frames if video failed
            blank = np.zeros((self.face_cropper.img_size, self.face_cropper.img_size, 3), dtype=np.uint8)
            x = torch.stack([self.transform(blank) for _ in range(self.frames_per_video)], dim=0)
        else:
            faces = [self.face_cropper.crop_face(f) for f in frames]  # RGB
            x = torch.stack([self.transform(im) for im in faces], dim=0)  # (T,C,H,W)

        if self.labels is None:
            return x, os.path.basename(vp)

        y = int(self.labels[idx])
        return x, y


# Model: CNN per-frame + temporal attention pooling
class TemporalAttnPool(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, 1)
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: (B, T, D)
        returns: (B, D)
        """
        w = self.attn(feats)          # (B,T,1)
        w = torch.softmax(w, dim=1)   # (B,T,1)
        return (feats * w).sum(dim=1)


class DeepfakeNet(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int = 2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool="")
        # Determine feature dim via a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.backbone(dummy)
            if feat.ndim == 4:
                feat = feat.mean(dim=[2, 3])
            feat_dim = feat.shape[-1]

        self.pool = TemporalAttnPool(feat_dim)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feat = self.backbone(x)
        if feat.ndim == 4:
            feat = feat.mean(dim=[2, 3])
        feat = feat.view(b, t, -1)  # (B,T,D)
        vid_feat = self.pool(feat)  # (B,D)
        logits = self.head(vid_feat)
        return logits


# Loss: label smoothing CE
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


# Metrics
@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()


# Training / Validation
def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, cfg: CFG):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if cfg.amp and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            if cfg.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits.detach(), y)

    return running_loss / max(1, len(loader)), running_acc / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    for x, y in tqdm(loader, desc="valid", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits, y)

    return running_loss / max(1, len(loader)), running_acc / max(1, len(loader))


def build_train_df(cfg: CFG) -> pd.DataFrame:
    real = list_videos(cfg.train_real_dir)
    fake = list_videos(cfg.train_fake_dir)

    df = pd.DataFrame({
        "path": real + fake,
        "label": [0] * len(real) + [1] * len(fake),  # 0=real, 1=fake
    })
    df["filename"] = df["path"].apply(lambda p: os.path.basename(p))
    return df


def run_train(cfg: CFG):
    os.makedirs(cfg.out_dir, exist_ok=True)
    seed_everything(cfg.seed)

    df = build_train_df(cfg)
    if len(df) == 0:
        raise RuntimeError("No training videos found. Check videos/train/real and videos/train/fake")

    # Stratified split
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    folds = list(skf.split(df["path"], df["label"]))
    tr_idx, va_idx = folds[cfg.fold_to_train]

    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_va = df.iloc[va_idx].reset_index(drop=True)

    device = cfg.device

    face_cropper = FaceCropper(device=device, img_size=cfg.img_size, margin=cfg.face_margin, min_face_size=cfg.min_face_size)
    tfm_train = build_transforms(cfg.img_size, train=True)
    tfm_valid = build_transforms(cfg.img_size, train=False)

    ds_tr = VideoFaceDataset(
        video_paths=df_tr["path"].tolist(),
        labels=df_tr["label"].tolist(),
        face_cropper=face_cropper,
        frames_per_video=cfg.frames_per_video,
        transform=tfm_train,
        is_train=True
    )
    ds_va = VideoFaceDataset(
        video_paths=df_va["path"].tolist(),
        labels=df_va["label"].tolist(),
        face_cropper=face_cropper,
        frames_per_video=cfg.frames_per_video,
        transform=tfm_valid,
        is_train=False
    )

    # Weighted sampler helps if you ever have imbalance
    class_counts = df_tr["label"].value_counts().to_dict()
    weights = []
    for y in df_tr["label"].tolist():
        weights.append(1.0 / class_counts[int(y)])
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = DeepfakeNet(cfg.backbone).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.startswith("cuda")))
    loss_fn = LabelSmoothingCE(cfg.label_smoothing)

    best_va_acc = -1.0
    best_path = cfg.model_path

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, dl_tr, optimizer, scaler, loss_fn, device, cfg)
        va_loss, va_acc = validate(model, dl_va, loss_fn, device)

        print(f"[Epoch {epoch:02d}/{cfg.epochs}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} valid_loss={va_loss:.4f} valid_acc={va_acc:.4f}")

        if va_acc > best_va_acc:
            best_va_acc = va_acc
            torch.save({
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "best_valid_acc": best_va_acc,
                "epoch": epoch,
            }, best_path)
            print(f"  -> saved best to {best_path} (valid_acc={best_va_acc:.4f})")

    print(f"Done. Best valid acc: {best_va_acc:.4f}")
    print(f"Best model at: {best_path}")


@torch.no_grad()
def predict_video(model: nn.Module, x: torch.Tensor, device: str) -> Tuple[int, float]:
    """
    x: (1,T,C,H,W)
    returns predicted_label_int, probability(max softmax)
    """
    model.eval()
    x = x.to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    pred = int(torch.argmax(probs).item())
    conf = float(probs[pred].item())
    return pred, conf


def run_predict(cfg: CFG, model_path: Optional[str] = None):
    os.makedirs(cfg.out_dir, exist_ok=True)
    seed_everything(cfg.seed)

    device = cfg.device
    mp = model_path or cfg.model_path
    if not os.path.exists(mp):
        raise RuntimeError(f"Model not found at {mp}. Train first or pass --model_path")

    ckpt = torch.load(mp, map_location="cpu")
    model = DeepfakeNet(cfg.backbone).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    face_cropper = FaceCropper(device=device, img_size=cfg.img_size, margin=cfg.face_margin, min_face_size=cfg.min_face_size)
    tfm = build_transforms(cfg.img_size, train=False)

    test_videos = list_videos(cfg.test_dir)
    if len(test_videos) == 0:
        raise RuntimeError("No test videos found in videos/test")

    rows = []
    for vp in tqdm(test_videos, desc="predict"):
        frames = safe_read_video_frames_cv2(vp, cfg.frames_per_video)
        if len(frames) == 0:
            blank = np.zeros((cfg.img_size, cfg.img_size, 3), dtype=np.uint8)
            x = torch.stack([tfm(blank) for _ in range(cfg.frames_per_video)], dim=0).unsqueeze(0)
        else:
            faces = [face_cropper.crop_face(f) for f in frames]
            x = torch.stack([tfm(im) for im in faces], dim=0).unsqueeze(0)

        pred, conf = predict_video(model, x, device)
        label = "real" if pred == 0 else "fake"

        rows.append({
            "filename": os.path.basename(vp),
            "label": label,
            "probability": conf
        })

    out_path = cfg.test_csv_path
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def run_finetune_full(cfg: CFG, resume_path: str, finetune_epochs: int):
    os.makedirs(cfg.out_dir, exist_ok=True)
    seed_everything(cfg.seed)

    df = build_train_df(cfg)
    if len(df) == 0:
        raise RuntimeError("No training videos found. Check videos/train/real and videos/train/fake")

    device = cfg.device

    face_cropper = FaceCropper(
        device=device,
        img_size=cfg.img_size,
        margin=cfg.face_margin,
        min_face_size=cfg.min_face_size
    )
    tfm_train = build_transforms(cfg.img_size, train=True)

    ds = VideoFaceDataset(
        video_paths=df["path"].tolist(),
        labels=df["label"].tolist(),
        face_cropper=face_cropper,
        frames_per_video=cfg.frames_per_video,
        transform=tfm_train,
        is_train=True
    )

    # Keep sampler (balanced). If your dataset is already perfectly balanced, it's still fine.
    class_counts = df["label"].value_counts().to_dict()
    weights = [1.0 / class_counts[int(y)] for y in df["label"].tolist()]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )

    model = DeepfakeNet(cfg.backbone).to(device)

    if not os.path.exists(resume_path):
        raise RuntimeError(f"Resume checkpoint not found: {resume_path}")

    ckpt = torch.load(resume_path, map_location="cpu")

    # load weights
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        # if you ever saved raw state_dict
        model.load_state_dict(ckpt, strict=True)

    print(f"Loaded checkpoint: {resume_path}")

    # Lower LR for fine-tuning usually helps
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr * 0.2, weight_decay=cfg.weight_decay)

    # Replace deprecated scaler call (optional but recommended)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.startswith("cuda")))

    loss_fn = LabelSmoothingCE(cfg.label_smoothing)

    for epoch in range(1, finetune_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, dl, optimizer, scaler, loss_fn, device, cfg)
        print(f"[Finetune {epoch:02d}/{finetune_epochs}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}")

        # Save each epoch (or only last — your choice)
        save_path = os.path.join(cfg.out_dir, f"model_finetune_ep{epoch:02d}.pt")
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "epoch": epoch}, save_path)
        print(f"  -> saved: {save_path}")

    # Also save a final convenient name
    final_path = os.path.join(cfg.out_dir, "model_finetuned_final.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "epoch": finetune_epochs}, final_path)
    print(f"Saved final finetuned model: {final_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser("Deepfake detection pipeline (face-based video)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data_root", default="videos")
    p_train.add_argument("--epochs", type=int, default=CFG.epochs)
    p_train.add_argument("--batch_size", type=int, default=CFG.batch_size)
    p_train.add_argument("--frames", type=int, default=CFG.frames_per_video)
    p_train.add_argument("--img_size", type=int, default=CFG.img_size)
    p_train.add_argument("--backbone", default=CFG.backbone)
    p_train.add_argument("--fold", type=int, default=CFG.fold_to_train)
    p_train.add_argument("--out_dir", default=CFG.out_dir)
    p_train.add_argument("--resume", default="", help="path to checkpoint .pt to resume/fine-tune from")
    p_train.add_argument("--finetune_epochs", type=int, default=0,
                         help="if >0, train on full data for this many epochs (no val)")

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--data_root", default="videos")
    p_pred.add_argument("--model_path", default=CFG.model_path)
    p_pred.add_argument("--frames", type=int, default=CFG.frames_per_video)
    p_pred.add_argument("--img_size", type=int, default=CFG.img_size)
    p_pred.add_argument("--backbone", default=CFG.backbone)
    p_pred.add_argument("--out_dir", default=CFG.out_dir)
    p_pred.add_argument("--csv_out", default=CFG.test_csv_path)

    args = parser.parse_args()

    cfg = CFG()
    cfg.data_root = args.data_root
    cfg.train_real_dir = os.path.join(args.data_root, "train", "real")
    cfg.train_fake_dir = os.path.join(args.data_root, "train", "fake")
    cfg.test_dir = os.path.join(args.data_root, "test")

    if args.cmd == "train":
        cfg.epochs = args.epochs
        cfg.batch_size = args.batch_size
        cfg.frames_per_video = args.frames
        cfg.img_size = args.img_size
        cfg.backbone = args.backbone
        cfg.fold_to_train = args.fold
        cfg.out_dir = args.out_dir
        cfg.model_path = os.path.join(cfg.out_dir, "model_best.pt")
        if args.finetune_epochs and args.finetune_epochs > 0:
            if not args.resume:
                raise RuntimeError("For --finetune_epochs, you must pass --resume path/to/model.pt")
            run_finetune_full(cfg, resume_path=args.resume, finetune_epochs=args.finetune_epochs)
        else:
            run_train(cfg)

    elif args.cmd == "predict":
        cfg.frames_per_video = args.frames
        cfg.img_size = args.img_size
        cfg.backbone = args.backbone
        cfg.out_dir = args.out_dir
        cfg.model_path = args.model_path
        cfg.test_csv_path = args.csv_out
        run_predict(cfg, model_path=args.model_path)


if __name__ == "__main__":
    main()