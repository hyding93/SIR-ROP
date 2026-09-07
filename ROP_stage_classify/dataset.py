import os
import random
import tempfile

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

cv2.setNumThreads(0)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

FUSION_LAMBDA = 0.5
RIDGE_KERNEL_SIZE = 25
TOP_HAT_WEIGHT = 0.3
CACHE_VERSION = "v1"


def generate_ridge_enhanced(
    image: Image.Image,
    kernel_size: int = RIDGE_KERNEL_SIZE,
    clip_limit: float = 2.0,
    tile_size: int = 8,
    top_hat_weight: float = TOP_HAT_WEIGHT,
    highlight_threshold: int = 240,
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    normalized = cv2.normalize(
        bgr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)

    saturated = gray > highlight_threshold
    median = cv2.medianBlur(gray, 5)
    gray = np.where(saturated, median, gray).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )
    enhanced = clahe.apply(gray)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    top_hat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
    ridge = cv2.addWeighted(enhanced, 1.0, top_hat, top_hat_weight, 0)

    return Image.fromarray(cv2.cvtColor(ridge, cv2.COLOR_GRAY2RGB))


def fuse_ridge_luminance(
    original: Image.Image,
    ridge: Image.Image,
    lam: float = FUSION_LAMBDA,
) -> Image.Image:
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    original_rgb = np.asarray(original.convert("RGB"))
    ridge_rgb = np.asarray(ridge.convert("RGB"))
    if ridge_rgb.shape[:2] != original_rgb.shape[:2]:
        raise ValueError(
            f"for aligned fusion, got {original_rgb.shape[:2]} and "
            f"{ridge_rgb.shape[:2]}."
        )

    ycrcb = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[..., 0].astype(np.float32)
    ridge_gray = cv2.cvtColor(ridge_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    ridge_std = float(ridge_gray.std()) + 1e-6
    y_std = float(y.std()) + 1e-6
    ridge_matched = (
        (ridge_gray - ridge_gray.mean()) * (y_std / ridge_std) + y.mean()
    )
    ridge_matched = np.clip(ridge_matched, 0, 255)

    y_fused = (1.0 - lam) * y + lam * ridge_matched
    fused_ycrcb = ycrcb.copy()
    fused_ycrcb[..., 0] = np.clip(y_fused, 0, 255).astype(np.uint8)
    fused_rgb = cv2.cvtColor(fused_ycrcb, cv2.COLOR_YCrCb2RGB)
    return Image.fromarray(fused_rgb)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


def _infer_input_size(shared_geometric, fallback: int = 512) -> int:
    for transform in getattr(shared_geometric, "transforms", []):
        if isinstance(transform, transforms.Resize):
            size = transform.size
            if isinstance(size, int):
                return int(size)
            if len(size) == 2 and int(size[0]) == int(size[1]):
                return int(size[0])
    return fallback


def _cache_tag(input_size: int, fusion_lambda: float) -> str:
    """Encode deterministic preprocessing settings into the cache directory."""
    tw = int(round(TOP_HAT_WEIGHT * 100))
    lam = int(round(fusion_lambda * 100))
    return (
        f".fused_cache_{input_size}_k{RIDGE_KERNEL_SIZE}_"
        f"tw{tw:03d}_lam{lam:03d}_{CACHE_VERSION}"
    )


class FusedImageFolder(torch.utils.data.Dataset):
    def __init__(
        self,
        orig_root: str,
        mask_root: str,
        shared_geometric=None,
        fused_color_only=None,
        mask_geometric=None,
        fusion_lambda: float = FUSION_LAMBDA,
        cache_root: str = None,
        input_size: int = None,
    ):
        self.orig_root = orig_root
        self.mask_root = mask_root
        self.shared_geometric = shared_geometric
        self.fused_color_only = fused_color_only
        self.mask_geometric = mask_geometric
        self.fusion_lambda = fusion_lambda
        self.input_size = (
            int(input_size)
            if input_size is not None
            else _infer_input_size(shared_geometric)
        )

        if cache_root is None:
            cache_root = os.path.join(
                os.path.dirname(os.path.normpath(orig_root)),
                _cache_tag(self.input_size, self.fusion_lambda),
            )
        self.cache_root = cache_root

        self.orig_dataset = datasets.ImageFolder(orig_root)
        self.classes = self.orig_dataset.classes
        self.class_to_idx = self.orig_dataset.class_to_idx
        self.samples = self.orig_dataset.samples
        self.targets = self.orig_dataset.targets

    def _cache_path(self, rel_path: str) -> str:
        base, _ = os.path.splitext(rel_path)
        return os.path.join(self.cache_root, base + ".png")

    def _read_valid_cache(self, cache_path: str, orig_path: str):
        """Return cached RGB image when valid; otherwise return None."""
        if not os.path.exists(cache_path):
            return None
        try:
            if os.path.getmtime(cache_path) < os.path.getmtime(orig_path):
                return None
        except OSError:
            return None

        try:
            with Image.open(cache_path) as cached:
                cached = cached.convert("RGB")
                if cached.size != (self.input_size, self.input_size):
                    return None
                return cached.copy()
        except (OSError, ValueError):
            return None

    @staticmethod
    def _atomic_save_png(image: Image.Image, cache_path: str) -> None:
        """Save cache without exposing partially-written files to workers."""
        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            prefix=".fused_",
            suffix=".tmp.png",
            dir=cache_dir,
        )
        os.close(fd)
        try:
            image.save(tmp_path, format="PNG", compress_level=1)
            os.replace(tmp_path, cache_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _get_or_create_fused(self, orig_path: str, rel_path: str) -> Image.Image:
        cache_path = self._cache_path(rel_path)
        cached = self._read_valid_cache(cache_path, orig_path)
        if cached is not None:
            return cached

        with Image.open(orig_path) as image_file:
            image = image_file.convert("RGB")
            image = image.resize(
                (self.input_size, self.input_size),
                resample=Image.Resampling.BICUBIC,
            )

        ridge = generate_ridge_enhanced(
            image,
            kernel_size=RIDGE_KERNEL_SIZE,
            top_hat_weight=TOP_HAT_WEIGHT,
        )
        fused = fuse_ridge_luminance(
            image,
            ridge,
            lam=self.fusion_lambda,
        )

        self._atomic_save_png(fused, cache_path)
        return fused

    def __getitem__(self, index):
        orig_path, target = self.samples[index]
        rel_path = os.path.relpath(orig_path, self.orig_root)
        base, _ = os.path.splitext(rel_path)
        mask_path = os.path.join(self.mask_root, base + ".png")

        fused = self._get_or_create_fused(orig_path, rel_path)

        if os.path.exists(mask_path):
            with Image.open(mask_path) as mask_file:
                mask_image = mask_file.convert("L").copy()
        else:
            mask_image = Image.new("L", fused.size, 0)

        if self.shared_geometric is not None:
            seed = random.randint(0, 2**32 - 1)

            _seed_all(seed)
            fused_geo = self.shared_geometric(fused)
            _seed_all(seed)
            mask_tensor = self.mask_geometric(mask_image)

            color_seed = random.randint(0, 2**32 - 1)
            _seed_all(color_seed)
            fused_tensor = self.fused_color_only(fused_geo)
        else:
            fused_tensor = transforms.ToTensor()(fused)
            mask_tensor = transforms.ToTensor()(mask_image)

        return fused_tensor, mask_tensor, target, os.path.basename(rel_path)

    def __len__(self):
        return len(self.samples)


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    input_size = args.input_size

    if is_train == "train":
        shared_geometric = transforms.Compose([
            transforms.Resize(
                (input_size, input_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(
                degrees=(-180, 180),
                translate=(0.15, 0.15),
                shear=(-10, 10, -10, 10),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
        ])
        mask_geometric = transforms.Compose([
            transforms.Resize(
                (input_size, input_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(
                degrees=(-180, 180),
                translate=(0.15, 0.15),
                shear=(-10, 10, -10, 10),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ])
        fused_color_only = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.01
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        shared_geometric = transforms.Compose([
            transforms.Resize(
                (input_size, input_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        ])
        mask_geometric = transforms.Compose([
            transforms.Resize(
                (input_size, input_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ])
        fused_color_only = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    return shared_geometric, fused_color_only, mask_geometric


def build_dataset(is_train, args):
    shared_geometric, fused_color_only, mask_geometric = build_transform(
        is_train, args
    )

    orig_root = os.path.join(args.data_path, is_train)
    mask_root = os.path.join(args.mask_data_path, is_train)
    cache_base = getattr(args, "fused_cache_path", None)
    if not cache_base:
        cache_base = os.path.join(
            args.data_path,
            _cache_tag(args.input_size, FUSION_LAMBDA),
        )
    cache_root = os.path.join(cache_base, is_train)

    print(f"Dataset mode: {is_train}")
    print(
        "Single fused RGB input: lazy-cache ridge + luminance fusion "
        f"lambda={FUSION_LAMBDA}"
    )
    print(
        f"Ridge: no circular mask, kernel={RIDGE_KERNEL_SIZE}x{RIDGE_KERNEL_SIZE}, "
        f"top-hat weight={TOP_HAT_WEIGHT}"
    )
    print(f"Fused cache: {cache_root}")

    dataset = FusedImageFolder(
        orig_root=orig_root,
        mask_root=mask_root,
        shared_geometric=shared_geometric,
        fused_color_only=fused_color_only,
        mask_geometric=mask_geometric,
        fusion_lambda=FUSION_LAMBDA,
        cache_root=cache_root,
        input_size=args.input_size,
    )
    return dataset
