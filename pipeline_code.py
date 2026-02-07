"""
deepfake_genconvit_singlefile.py

A single-file deepfake detection pipeline that merges:
- config (no YAML)
- data loader + albumentations augmentation (your loader.py)
- face extraction + video inference helpers
- GenConViT models (ED / VAE / both) with collision-free class names

NOTES:
1) You still need to paste in your real HybridEmbed implementation if you used a custom one.
   If you used timm's built-in patch_embed only, you can remove HybridEmbed usage.
2) Requirements: torch, torchvision, timm, albumentations, opencv-python, face_recognition, dlib, decord, pillow, tqdm
"""

import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

# torchvision / timm
from torchvision import transforms, datasets
import timm
from timm import create_model

# video
from decord import VideoReader, cpu

# face
import dlib
import face_recognition

# albumentations
from albumentations import (
    Compose,
    RandomRotate90,
    Transpose,
    HorizontalFlip,
    VerticalFlip,
    OneOf,
    GaussNoise,
    ShiftScaleRotate,
    CLAHE,
    Sharpen,
    Emboss,
    RandomBrightnessContrast,
    HueSaturationValue,
)

# ---------------------------------------------------------------------
# 0) DEVICE + CONFIG (replaces YAML)
# ---------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "model": {
        "backbone": "convnext_tiny",
        "embedder": "swin_tiny_patch4_window7_224",
        # IMPORTANT: your YAML had latent_dims: 12544 which equals 256*7*7
        "latent_dims": 12544,
    },
    "batch_size": 32,
    "epoch": 1,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "num_classes": 2,
    "img_size": 224,
    "min_val_loss": 10000,
}


def load_config():
    return CONFIG


# ---------------------------------------------------------------------
# 1) AUGMENTATION + NORMALIZATION (your loader.py merged)
# ---------------------------------------------------------------------
def strong_aug(p=0.5):
    return Compose(
        [
            RandomRotate90(p=0.2),
            Transpose(p=0.2),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            OneOf([GaussNoise()], p=0.2),
            ShiftScaleRotate(p=0.2),
            OneOf(
                [
                    CLAHE(clip_limit=2),
                    Sharpen(),
                    Emboss(),
                    RandomBrightnessContrast(),
                ],
                p=0.2,
            ),
            HueSaturationValue(p=0.2),
        ],
        p=p,
    )


def augment(aug, image):
    return aug(image=image)["image"]


class AlbumentationsAugment:
    """Wrap albumentations so it can be used inside torchvision transforms pipeline."""

    def __call__(self, img: Image.Image) -> Image.Image:
        aug = strong_aug(p=0.9)
        arr = np.array(img)
        out = augment(aug, arr)
        return Image.fromarray(out)


def normalize_data():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return {
        "train": transforms.Compose(
            [AlbumentationsAugment(), transforms.ToTensor(), transforms.Normalize(mean, std)]
        ),
        "valid": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        ),
        "test": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        ),
        # For video pipeline we normalize per-frame tensor already in CHW
        "vid": transforms.Compose([transforms.Normalize(mean, std)]),
    }


def load_data(data_dir="sample/", batch_size=4, num_workers=0):
    image_datasets = {
        split: datasets.ImageFolder(os.path.join(data_dir, split), normalize_data()[split])
        for split in ["train", "valid", "test"]
    }
    dataset_sizes = {split: len(image_datasets[split]) for split in ["train", "valid", "test"]}

    train_loader = torch.utils.data.DataLoader(
        image_datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        image_datasets["valid"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        image_datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    dataloaders = {"train": train_loader, "validation": valid_loader, "test": test_loader}
    return dataloaders, dataset_sizes


def load_checkpoint(model, optimizer, filename=None):
    start_epoch = 0
    log_loss = 0
    if filename and os.path.isfile(filename):
        print(f"=> loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location="cpu")
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        log_loss = checkpoint["min_loss"]
        print(f"=> loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
    else:
        print(f"=> no checkpoint found at '{filename}'")

    return model, optimizer, start_epoch, log_loss


# ---------------------------------------------------------------------
# 2) VIDEO + FACE EXTRACTION PIPELINE
# ---------------------------------------------------------------------
def face_recognize_frames(frames_rgb):
    """
    frames_rgb: list/array of RGB frames (H,W,3) uint8
    returns: (faces_np, count) where faces_np is (T,224,224,3) RGB
    """
    temp_face = np.zeros((len(frames_rgb), 224, 224, 3), dtype=np.uint8)
    count = 0
    model_name = "cnn" if dlib.DLIB_USE_CUDA else "hog"

    for _, frame in tqdm(enumerate(frames_rgb), total=len(frames_rgb), leave=False):
        # face_recognition expects RGB by default; your old code converted RGB->BGR then used face_locations.
        # We'll keep it consistent with your original behavior:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        face_locations = face_recognition.face_locations(
            frame_bgr, number_of_times_to_upsample=0, model=model_name
        )

        for (top, right, bottom, left) in face_locations:
            if count >= len(frames_rgb):
                break
            face_bgr = frame_bgr[top:bottom, left:right]
            face_bgr = cv2.resize(face_bgr, (224, 224), interpolation=cv2.INTER_AREA)
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            temp_face[count] = face_rgb
            count += 1

    return ([], 0) if count == 0 else (temp_face[:count], count)


def preprocess_face_frames(face_frames_np):
    """
    face_frames_np: (T,224,224,3) uint8 RGB
    returns: (T,3,224,224) float tensor on device, normalized
    """
    df_tensor = torch.tensor(face_frames_np, device=device).float()  # T,H,W,C
    df_tensor = df_tensor.permute((0, 3, 1, 2))  # T,C,H,W

    vid_norm = normalize_data()["vid"]
    for i in range(len(df_tensor)):
        df_tensor[i] = vid_norm(df_tensor[i] / 255.0)

    return df_tensor


def extract_frames(video_file, num_frames=15):
    vr = VideoReader(video_file, ctx=cpu(0))
    total_frames = len(vr)

    if num_frames == -1:
        indices = np.arange(total_frames).astype(int)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    return vr.get_batch(indices).asnumpy()  # RGB uint8


def deepfake_tensor_from_video(video_path, num_frames=15):
    frames = extract_frames(video_path, num_frames)
    faces, count = face_recognize_frames(frames)
    return preprocess_face_frames(faces) if count > 0 else []


def deepfake_tensor_from_folder(folder_path, num_frames=15):
    img_list = glob.glob(os.path.join(folder_path, "*"))
    imgs = []
    for f in img_list:
        try:
            im = Image.open(f).convert("RGB")
            imgs.append(np.asarray(im))
        except Exception:
            pass

    faces, count = face_recognize_frames(imgs[:num_frames])
    return preprocess_face_frames(faces) if count > 0 else []


def is_video(path):
    return os.path.isfile(path) and path.lower().endswith((".avi", ".mp4", ".mpg", ".mpeg", ".mov"))


def is_video_folder(path):
    img_list = glob.glob(os.path.join(path, "*"))
    if len(img_list) < 1:
        return False
    return img_list[0].lower().endswith((".png", ".jpeg", ".jpg"))


# ---------------------------------------------------------------------
# 3) PREDICTION HELPERS
# ---------------------------------------------------------------------
@torch.no_grad()
def pred_vid(df_tensor, model):
    # model(df_tensor) -> logits; apply sigmoid then reduce
    return max_prediction_value(torch.sigmoid(model(df_tensor).squeeze()))


def max_prediction_value(y_pred):
    mean_val = torch.mean(y_pred, dim=0)
    return (
        torch.argmax(mean_val).item(),
        mean_val[0].item() if mean_val[0] > mean_val[1] else abs(1 - mean_val[1]).item(),
    )


def real_or_fake(prediction):
    return {0: "REAL", 1: "FAKE"}[prediction ^ 1]


def set_result():
    return {
        "video": {"name": [], "pred": [], "klass": [], "pred_label": [], "correct_label": []}
    }


def store_result(result, filename, y, y_val, klass, correct_label=None, compression=None):
    result["video"]["name"].append(filename)
    result["video"]["pred"].append(y_val)
    result["video"]["klass"].append(str(klass).lower())
    result["video"]["pred_label"].append(real_or_fake(y))

    if correct_label is not None:
        result["video"]["correct_label"].append(correct_label)
    if compression is not None:
        result["video"].setdefault("compression", [])
        result["video"]["compression"].append(compression)

    return result


# ---------------------------------------------------------------------
# 4) MODEL EMBEDDER (HybridEmbed)
# ---------------------------------------------------------------------
class HybridEmbed(nn.Module):
    """
    IMPORTANT:
    Replace this with your REAL HybridEmbed from your project if it was custom.

    If your original code depends on a specific HybridEmbed behavior, leaving this as-is
    will break or degrade performance.
    """

    def __init__(self, embedder: nn.Module, img_size: int, embed_dim: int):
        super().__init__()
        self.embedder = embedder
        self.img_size = img_size
        self.embed_dim = embed_dim

    def forward(self, x):
        raise NotImplementedError("Paste your real HybridEmbed implementation here.")


# ---------------------------------------------------------------------
# 5) GENCONVIT MODELS (renamed to avoid collisions)
# ---------------------------------------------------------------------
# -------------------- VAE branch --------------------
class VAEFeatEncoder(nn.Module):
    def __init__(self, latent_dims: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.latent_dims = latent_dims

        # assumes input 224 -> feature map 14x14
        self.mu = nn.Linear(128 * 14 * 14, latent_dims)
        self.logvar = nn.Linear(128 * 14 * 14, latent_dims)

        self.kl = 0.0
        self.kl_weight = 0.5

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)

        mu = self.mu(x)
        logvar = self.logvar(x)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu

        self.kl = self.kl_weight * torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1),
            dim=0,
        )
        return z


class VAEFeatDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # latent_dims expected to be 256*7*7 == 12544 so we can unflatten to (256,7,7)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 7, 7))
        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 2, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 2, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 2, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 3, 2, 2),
            nn.LeakyReLU(),
        )

    def forward(self, z):
        x = self.unflatten(z)
        return self.features(x)


class DeepfakeGenConViT_VAE(nn.Module):
    def __init__(self, config, pretrained=True):
        super().__init__()
        latent_dims = int(config["model"]["latent_dims"])
        self.encoder = VAEFeatEncoder(latent_dims)
        self.decoder = VAEFeatDecoder()

        self.embedder = create_model(config["model"]["embedder"], pretrained=pretrained)
        self.backbone = create_model(
            config["model"]["backbone"],
            pretrained=pretrained,
            num_classes=1000,
            drop_path_rate=0,
            head_init_scale=1.0,
        )
        self.backbone.patch_embed = HybridEmbed(self.embedder, img_size=config["img_size"], embed_dim=768)

        self.num_feature = self.backbone.head.fc.out_features * 2
        self.fc = nn.Linear(self.num_feature, self.num_feature // 4)
        self.fc2 = nn.Linear(self.num_feature // 4, config["num_classes"])
        self.relu = nn.ReLU()
        self.resize = transforms.Resize((224, 224), antialias=True)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)

        x1 = self.backbone(x)
        x2 = self.backbone(x_hat)
        feat = torch.cat((x1, x2), dim=1)

        logits = self.fc2(self.relu(self.fc(self.relu(feat))))
        return logits, self.resize(x_hat)


# -------------------- ED branch --------------------
class EDFEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.features(x)


class EDFEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 2, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 2, 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.features(x)


class DeepfakeGenConViT_ED(nn.Module):
    def __init__(self, config, pretrained=True):
        super().__init__()
        self.encoder = EDFEEncoder()
        self.decoder = EDFEDecoder()

        self.backbone = timm.create_model(config["model"]["backbone"], pretrained=pretrained)
        self.embedder = timm.create_model(config["model"]["embedder"], pretrained=pretrained)
        self.backbone.patch_embed = HybridEmbed(self.embedder, img_size=config["img_size"], embed_dim=768)

        self.num_features = self.backbone.head.fc.out_features * 2
        self.fc = nn.Linear(self.num_features, self.num_features // 4)
        self.fc2 = nn.Linear(self.num_features // 4, 2)
        self.act = nn.GELU()

    def forward(self, images):
        enc = self.encoder(images)
        dec = self.decoder(enc)

        x1 = self.backbone(dec)
        x2 = self.backbone(images)

        feat = torch.cat((x1, x2), dim=1)
        logits = self.fc2(self.act(self.fc(self.act(feat))))
        return logits


# -------------------- Wrapper --------------------
class DeepfakeGenConViT(nn.Module):
    """
    net:
      - "ed"  : ED only
      - "vae" : VAE only
      - "both": concatenates logits (as your original did)
    """

    def __init__(self, config, net="ed", ed_weight=None, vae_weight=None, fp16=False):
        super().__init__()
        self.net = net
        self.fp16 = fp16

        if net == "ed":
            self.model_ed = DeepfakeGenConViT_ED(config)
            if ed_weight:
                chk = torch.load(f"weight/{ed_weight}.pth", map_location="cpu")
                self.model_ed.load_state_dict(chk.get("state_dict", chk))
            self.model_ed.eval()
            if fp16:
                self.model_ed.half()

        elif net == "vae":
            self.model_vae = DeepfakeGenConViT_VAE(config)
            if vae_weight:
                chk = torch.load(f"weight/{vae_weight}.pth", map_location="cpu")
                self.model_vae.load_state_dict(chk.get("state_dict", chk))
            self.model_vae.eval()
            if fp16:
                self.model_vae.half()

        else:
            self.model_ed = DeepfakeGenConViT_ED(config)
            self.model_vae = DeepfakeGenConViT_VAE(config)

            if ed_weight:
                chk = torch.load(f"weight/{ed_weight}.pth", map_location="cpu")
                self.model_ed.load_state_dict(chk.get("state_dict", chk))
            if vae_weight:
                chk = torch.load(f"weight/{vae_weight}.pth", map_location="cpu")
                self.model_vae.load_state_dict(chk.get("state_dict", chk))

            self.model_ed.eval()
            self.model_vae.eval()
            if fp16:
                self.model_ed.half()
                self.model_vae.half()

    def forward(self, x):
        if self.net == "ed":
            return self.model_ed(x)
        if self.net == "vae":
            logits, _ = self.model_vae(x)
            return logits

        x1 = self.model_ed(x)
        x2, _ = self.model_vae(x)
        return torch.cat((x1, x2), dim=0)


def load_deepfake_model(config, net="ed", ed_weight=None, vae_weight=None, fp16=False):
    model = DeepfakeGenConViT(config, net=net, ed_weight=ed_weight, vae_weight=vae_weight, fp16=fp16)
    model.to(device)
    model.eval()
    if fp16:
        model.half()
    return model


# ---------------------------------------------------------------------
# 6) QUICK DEMO USAGE
# ---------------------------------------------------------------------
if __name__ == "__main__":
    cfg = load_config()

    # 1) Load dataloaders
    # dataloaders, sizes = load_data(data_dir="sample/", batch_size=cfg["batch_size"])

    # 2) Load model (example: ED)
    # model = load_deepfake_model(cfg, net="ed", ed_weight="YOUR_ED_WEIGHT_NAME", fp16=False)

    # 3) Run on a video
    # video_path = "some_video.mp4"
    # x = deepfake_tensor_from_video(video_path, num_frames=15)
    # if isinstance(x, torch.Tensor) and len(x) > 0:
    #     y, score = pred_vid(x, model)
    #     print("Prediction:", real_or_fake(y), "Score:", score)
    pass