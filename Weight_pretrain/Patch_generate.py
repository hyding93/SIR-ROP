from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import morphology
from tqdm import tqdm

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
def _resolve_adaptive_px(
    patch_size: int,
    fixed_px: int,
    ratio: float,
    min_px: int,
    max_px: int,
) -> int:
    if fixed_px > 0:
        return int(fixed_px)
    value = int(round(float(patch_size) * float(ratio)))
    return int(np.clip(value, int(min_px), int(max_px)))


def _build_vessel_masks(
    vessel_mask: np.ndarray,
    core_radius: int,
    blend_radius: int,
    donor_margin: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vessel = (vessel_mask > 0).astype(np.uint8)
    if not np.any(vessel):
        z = np.zeros_like(vessel, dtype=np.uint8)
        return z, z.copy(), z.copy(), z.astype(np.float32)

    def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0:
            return mask.copy()
        k = 2 * int(radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return (cv2.dilate(mask * 255, kernel, iterations=1) > 0).astype(np.uint8)

    core = _dilate(vessel, core_radius)
    blend = _dilate(core, blend_radius)
    exclusion = _dilate(blend, donor_margin)

    alpha = np.zeros_like(vessel, dtype=np.float32)
    alpha[core > 0] = 1.0

    ring = (blend > 0) & (core == 0)
    if blend_radius > 0 and np.any(ring):
        dist_to_core = cv2.distanceTransform((core == 0).astype(np.uint8), cv2.DIST_L2, 5)
        ring_alpha = 1.0 - dist_to_core / (float(blend_radius) + 1.0)
        alpha[ring] = np.clip(ring_alpha[ring], 1e-3, 1.0 - 1e-3)

    return core, blend, exclusion, alpha


def remove_vessels_directional_texture(
    img_bgr,
    vessel_mask,
    dilate_px=0,
    sample_margin=0,
    sigma_low=5.0,
    sigma_tex=2.0,
    feather_px=0,
    step_min=2,
    step_max=32,
    dilate_ratio=0.02,
    dilate_min_px=3,
    dilate_max_px=12,
    feather_ratio=0.008,
    feather_min_px=2,
    feather_max_px=6,
    sample_margin_ratio=0.015,
    sample_margin_min_px=4,
    sample_margin_max_px=10,
    tangent_jitter_px=2,
    rng=None,
):
    img = img_bgr.astype(np.float32)
    vessel_mask = (vessel_mask > 0).astype(np.uint8)

    if vessel_mask.sum() == 0:
        return img_bgr.copy(), np.zeros_like(vessel_mask, dtype=np.uint8)

    h, w = vessel_mask.shape
    patch_size = min(h, w)
    core_radius = _resolve_adaptive_px(
        patch_size, dilate_px, dilate_ratio, dilate_min_px, dilate_max_px
    )
    blend_radius = _resolve_adaptive_px(
        patch_size, feather_px, feather_ratio, feather_min_px, feather_max_px
    )
    donor_margin = _resolve_adaptive_px(
        patch_size, sample_margin, sample_margin_ratio,
        sample_margin_min_px, sample_margin_max_px
    )

    core_mask, blend_mask, sample_exclusion, alpha2d = _build_vessel_masks(
        vessel_mask,
        core_radius=core_radius,
        blend_radius=blend_radius,
        donor_margin=donor_margin,
    )

    telea = cv2.inpaint(
        img_bgr,
        (core_mask * 255).astype(np.uint8),
        7,
        cv2.INPAINT_TELEA,
    ).astype(np.float32)

    telea_low = cv2.GaussianBlur(
        telea, (0, 0), sigmaX=sigma_low, sigmaY=sigma_low
    )

    texture_low = cv2.GaussianBlur(
        img, (0, 0), sigmaX=sigma_tex, sigmaY=sigma_tex
    )
    residual = img - texture_low

    recon_mask = blend_mask
    dist = cv2.distanceTransform(recon_mask, cv2.DIST_L2, 5)
    gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx_map = gx / magnitude
    ny_map = gy / magnitude

    ys, xs = np.where(recon_mask > 0)
    if xs.size == 0:
        return img_bgr.copy(), (core_mask * 255).astype(np.uint8)

    xs_float = xs.astype(np.float32)
    ys_float = ys.astype(np.float32)
    nx = nx_map[ys, xs]
    ny = ny_map[ys, xs]
    n = len(xs)

    plus_texture = np.zeros((n, 3), dtype=np.float32)
    minus_texture = np.zeros((n, 3), dtype=np.float32)
    plus_distance = np.zeros(n, dtype=np.float32)
    minus_distance = np.zeros(n, dtype=np.float32)
    found_plus = np.zeros(n, dtype=bool)
    found_minus = np.zeros(n, dtype=bool)

    if rng is None:
        rng = np.random.default_rng(0)
    tangent_jitter_px = max(int(tangent_jitter_px), 0)
    if tangent_jitter_px > 0:
        jitter = rng.integers(-tangent_jitter_px, tangent_jitter_px + 1, size=n).astype(np.float32)
    else:
        jitter = np.zeros(n, dtype=np.float32)
    tx = -ny
    ty = nx

    for step in range(step_min, step_max + 1):
        if not found_plus.all():
            xx = np.rint(xs_float + nx * step + tx * jitter).astype(np.int32)
            yy = np.rint(ys_float + ny * step + ty * jitter).astype(np.int32)
            valid = (
                (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h) & (~found_plus)
            )
            idx = np.flatnonzero(valid)
            if idx.size:
                idx = idx[sample_exclusion[yy[idx], xx[idx]] == 0]
                if idx.size:
                    plus_texture[idx] = residual[yy[idx], xx[idx]]
                    plus_distance[idx] = float(step)
                    found_plus[idx] = True

        if not found_minus.all():
            xx = np.rint(xs_float - nx * step + tx * jitter).astype(np.int32)
            yy = np.rint(ys_float - ny * step + ty * jitter).astype(np.int32)
            valid = (
                (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h) & (~found_minus)
            )
            idx = np.flatnonzero(valid)
            if idx.size:
                idx = idx[sample_exclusion[yy[idx], xx[idx]] == 0]
                if idx.size:
                    minus_texture[idx] = residual[yy[idx], xx[idx]]
                    minus_distance[idx] = float(step)
                    found_minus[idx] = True

        if found_plus.all() and found_minus.all():
            break

    sampled_texture = np.zeros((n, 3), dtype=np.float32)
    both = found_plus & found_minus
    only_plus = found_plus & (~found_minus)
    only_minus = found_minus & (~found_plus)

    if np.any(both):
        denom = plus_distance[both] + minus_distance[both] + 1e-6
        w_plus = minus_distance[both] / denom
        w_minus = plus_distance[both] / denom
        norm = np.sqrt(w_plus * w_plus + w_minus * w_minus) + 1e-6
        sampled_texture[both] = (
            w_plus[:, None] * plus_texture[both]
            + w_minus[:, None] * minus_texture[both]
        ) / norm[:, None]

    if np.any(only_plus):
        sampled_texture[only_plus] = plus_texture[only_plus]
    if np.any(only_minus):
        sampled_texture[only_minus] = minus_texture[only_minus]

    fill_residual = np.zeros_like(img, dtype=np.float32)
    reliable_normal = (np.abs(nx) + np.abs(ny)) >= 1e-4
    usable = reliable_normal & (found_plus | found_minus)
    if np.any(usable):
        fill_residual[ys[usable], xs[usable]] = sampled_texture[usable]

    fill = np.clip(telea_low + fill_residual, 0, 255)
    alpha = alpha2d[..., None]
    output = img * (1.0 - alpha) + fill * alpha
    output = np.clip(output, 0, 255).astype(np.uint8)

    return output, (core_mask * 255).astype(np.uint8)


def erase_vessels_directional(
    patch_rgb: np.ndarray,
    vessel_mask2d: np.ndarray,
    dilate_px: int = 0,
    sample_margin: int = 0,
    sigma_low: float = 5.0,
    sigma_tex: float = 2.0,
    feather_px: float = 0,
    step_min: int = 2,
    step_max: int = 32,
    dilate_ratio: float = 0.02,
    dilate_min_px: int = 3,
    dilate_max_px: int = 12,
    feather_ratio: float = 0.008,
    feather_min_px: int = 2,
    feather_max_px: int = 6,
    sample_margin_ratio: float = 0.015,
    sample_margin_min_px: int = 4,
    sample_margin_max_px: int = 10,
    tangent_jitter_px: int = 2,
    rng=None,
) -> np.ndarray:
    patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
    out_bgr, _ = remove_vessels_directional_texture(
        patch_bgr,
        vessel_mask2d.astype(np.uint8),
        dilate_px=dilate_px,
        sample_margin=sample_margin,
        sigma_low=sigma_low,
        sigma_tex=sigma_tex,
        feather_px=feather_px,
        step_min=step_min,
        step_max=step_max,
        dilate_ratio=dilate_ratio,
        dilate_min_px=dilate_min_px,
        dilate_max_px=dilate_max_px,
        feather_ratio=feather_ratio,
        feather_min_px=feather_min_px,
        feather_max_px=feather_max_px,
        sample_margin_ratio=sample_margin_ratio,
        sample_margin_min_px=sample_margin_min_px,
        sample_margin_max_px=sample_margin_max_px,
        tangent_jitter_px=tangent_jitter_px,
        rng=rng,
    )
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
def creatMask(img: np.ndarray, threshold: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    if img is None:
        raise ValueError('Image is None')

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask0 = gray >= threshold
    elif img.ndim == 2:
        mask0 = img >= threshold
    else:
        raise ValueError(f'Unsupported image shape: {img.shape}')

    mask0 = np.uint8(mask0)
    contours, _ = cv2.findContours(mask0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    if contours:
        max_index = int(np.argmax([cv2.contourArea(c) for c in contours]))
        cv2.drawContours(mask, contours, max_index, 1, -1)

    result_img = img.copy()
    if img.ndim == 3:
        result_img[mask == 0] = (255, 255, 255)
    else:
        result_img[mask == 0] = 255

    return result_img, mask
def patchJudge(a, v, av, y, x, patch_h, patch_w, av_type=0, limit=100) -> bool:
    a_sum = int(np.sum(a[y:y + patch_h, x:x + patch_w]))
    v_sum = int(np.sum(v[y:y + patch_h, x:x + patch_w]))
    av_sum = int(np.sum(av[y:y + patch_h, x:x + patch_w]))

    if av_type == 0:
        return a_sum >= limit and v_sum == 0
    if av_type == 1:
        return v_sum >= limit and a_sum == 0
    if av_type == 2:
        if a_sum < limit or v_sum < limit or av_sum < 2 * limit:
            return False
        return (a_sum / max(v_sum, 1) < 4.0) and (v_sum / max(a_sum, 1) < 4.0)
    if av_type == 3:
        return av_sum == 0
    return True


def _safe_randint_inclusive(rng: np.random.Generator, low: int, high: int) -> Optional[int]:
    if high < low:
        return None
    return int(rng.integers(low, high + 1))


def patch_select_for_RIP(
    rng: np.random.Generator,
    patch_size: int,
    Mask: np.ndarray,
    LabelA: np.ndarray,
    LabelV: np.ndarray,
    LabelVessel: np.ndarray,
    av_type: int = 0,
    max_attempts: int = 100,
):
    H, W = LabelVessel.shape
    if patch_size <= 0 or patch_size > H or patch_size > W:
        return 0, 0, patch_size, False

    if av_type == 3:
        non_vessel_points = np.argwhere(LabelVessel == 0)
        if non_vessel_points.size == 0:
            return 0, 0, patch_size, False

        half = patch_size // 2
        for _ in range(max_attempts):
            point_idx = int(rng.integers(0, len(non_vessel_points)))
            cy, cx = map(int, non_vessel_points[point_idx])

            y = int(np.clip(cy - half, 0, H - patch_size))
            x = int(np.clip(cx - half, 0, W - patch_size))

            if patchJudge(LabelA, LabelV, LabelVessel, y, x, patch_size, patch_size,
                          av_type=3, limit=100):
                return y, x, patch_size, True

        return 0, 0, patch_size, False

    skel = Mask[2, :, :]
    skels = np.argwhere(skel > 0)
    if skels.size == 0:
        return 0, 0, patch_size, False

    roi_y0 = roi_x0 = 0
    roi_y1, roi_x1 = H, W

    for attempt in range(max_attempts):
        if attempt % 50 == 0:
            dot_pix = int(rng.integers(0, len(skels)))
            y_axis, x_axis = map(int, skels[dot_pix])
            roi_x0 = max(0, x_axis - patch_size)
            roi_y0 = max(0, y_axis - patch_size)
            roi_x1 = min(x_axis + patch_size, W)
            roi_y1 = min(y_axis + patch_size, H)

        y = _safe_randint_inclusive(rng, roi_y0, roi_y1 - patch_size)
        x = _safe_randint_inclusive(rng, roi_x0, roi_x1 - patch_size)
        if y is None or x is None:
            continue

        if patchJudge(LabelA, LabelV, LabelVessel, y, x, patch_size, patch_size,
                      av_type=av_type, limit=100):
            return y, x, patch_size, True

    return 0, 0, patch_size, False


def check_overlap_iou(
    candidate: Tuple[int, int, int],
    selected_boxes: List[Tuple[int, int, int]],
    threshold: float = 0.3,
) -> bool:
    if not selected_boxes:
        return False

    x, y, s = candidate
    x2, y2 = x + s, y + s
    area1 = float(s * s)

    for sx, sy, ss in selected_boxes:
        sx2, sy2 = sx + ss, sy + ss
        inter_w = max(0, min(x2, sx2) - max(x, sx))
        inter_h = max(0, min(y2, sy2) - max(y, sy))
        inter = float(inter_w * inter_h)
        if inter <= 0:
            continue
        area2 = float(ss * ss)
        union = area1 + area2 - inter
        if union > 0 and (inter / union) >= threshold:
            return True
    return False

def _valid_blur_kernel(requested: int, h: int, w: int) -> int:
    k = min(requested, h if h % 2 == 1 else h - 1, w if w % 2 == 1 else w - 1)
    k = max(k, 1)
    if k % 2 == 0:
        k -= 1
    return max(k, 1)


def _local_stats_maps(img: np.ndarray, win: int) -> Tuple[np.ndarray, np.ndarray]:
    img_f = img.astype(np.float32)
    k = _valid_blur_kernel(win, img.shape[0], img.shape[1])
    local_mean = cv2.GaussianBlur(img_f, (k, k), 0)
    local_sq_mean = cv2.GaussianBlur(img_f * img_f, (k, k), 0)
    local_var = np.clip(local_sq_mean - local_mean * local_mean, 0, None)
    local_std = np.sqrt(local_var)
    return local_mean, local_std


def _resolve_structure_sigma(patch_size: int) -> float:
    return float(np.clip(float(patch_size) * 0.01, 1.5, 4.0))


def _robust_cap_local_std(std_map: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    std = std_map.astype(np.float32).copy()
    if std.ndim == 2:
        cap = float(np.percentile(std, percentile))
        return np.minimum(std, cap)
    for ch in range(std.shape[2]):
        cap = float(np.percentile(std[..., ch], percentile))
        std[..., ch] = np.minimum(std[..., ch], cap)
    return std


def _highpass_reference_for_noise(reference: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    ref = reference.astype(np.float32)
    low = cv2.GaussianBlur(ref, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if ref.ndim == 3 and ref.shape[2] == 1 and low.ndim == 2:
        low = low[..., None]
    return ref - low


def _spectral_shaped_noise(h: int, w: int, c: int, reference: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty((h, w, c), dtype=np.float32)
    for ch in range(c):
        ref = reference[..., ch].astype(np.float32)
        if ref.shape[:2] != (h, w):
            ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_AREA)
        ref_hp = _highpass_reference_for_noise(ref[..., None], sigma=2.0)[..., 0]
        ref_spec = np.fft.fft2(ref_hp - ref_hp.mean())
        mag = np.abs(ref_spec)
        mag_shift = np.fft.fftshift(mag)
        yy, xx = np.indices((h, w))
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        rr = np.rint(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)).astype(np.int32)
        radial_sum = np.bincount(rr.ravel(), weights=mag_shift.ravel())
        radial_n = np.bincount(rr.ravel())
        radial_mean = radial_sum / np.maximum(radial_n, 1)
        mag_iso = np.fft.ifftshift(radial_mean[rr])

        white = rng.standard_normal((h, w)).astype(np.float32)
        white_spec = np.fft.fft2(white)
        phase = white_spec / (np.abs(white_spec) + 1e-8)

        shaped = np.real(np.fft.ifft2(mag_iso * phase))
        std = shaped.std()
        shaped = (shaped - shaped.mean()) / (std + 1e-8)
        out[..., ch] = shaped
    return out


def _sample_overlay_rect(h: int, w: int, area_min: float, area_max: float,
                          rng: np.random.Generator) -> Tuple[int, int, int, int]:
    total_area = float(h * w)
    area_frac = float(rng.uniform(area_min, area_max))
    target_area = area_frac * total_area

    aspect = float(rng.uniform(0.5, 2.0))
    rect_w = int(round(np.sqrt(target_area * aspect)))
    rect_h = int(round(target_area / max(rect_w, 1)))

    rect_w = int(np.clip(rect_w, 1, w))
    rect_h = int(np.clip(rect_h, 1, h))

    x0 = int(rng.integers(0, w - rect_w + 1))
    y0 = int(rng.integers(0, h - rect_h + 1))
    return y0, x0, rect_h, rect_w



def _sample_irregular_overlay_mask(
    h: int,
    w: int,
    area_min: float,
    area_max: float,
    rng: np.random.Generator,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if valid_mask is None:
        valid = np.ones((h, w), dtype=bool)
    else:
        if valid_mask.shape != (h, w):
            raise ValueError('valid_mask shape must match (h, w)')
        valid = valid_mask > 0
    valid_count = int(valid.sum())
    if valid_count < 2:
        raise ValueError('valid_mask contains too few valid pixels')

    area_frac = float(rng.uniform(area_min, area_max))
    target = int(np.clip(round(area_frac * valid_count), 1, valid_count - 1))

    gh = max(6, min(24, h // 12))
    gw = max(6, min(24, w // 12))
    coarse = rng.standard_normal((gh, gw)).astype(np.float32)
    smooth = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    smooth = cv2.GaussianBlur(smooth, (0, 0), sigmaX=max(w, h) / 80.0)
    smooth = (smooth - smooth.mean()) / (smooth.std() + 1e-6)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    valid_ys, valid_xs = np.where(valid)
    cx = float(rng.uniform(valid_xs.min(), valid_xs.max() + 1))
    cy = float(rng.uniform(valid_ys.min(), valid_ys.max() + 1))
    sx = float(rng.uniform(0.28, 0.48) * w)
    sy = float(rng.uniform(0.28, 0.48) * h)
    local_bias = -(((xx - cx) / (sx + 1e-6)) ** 2 + ((yy - cy) / (sy + 1e-6)) ** 2)

    score = local_bias + 0.35 * smooth
    valid_scores = score[valid]
    kth = valid_scores.size - target
    threshold = np.partition(valid_scores, kth)[kth]
    mask = ((score >= threshold) & valid).astype(np.uint8)

    if int(mask.sum()) > target:
        ys, xs = np.where(mask > 0)
        vals = score[ys, xs]
        keep_idx = np.argpartition(vals, -target)[-target:]
        trimmed = np.zeros_like(mask)
        trimmed[ys[keep_idx], xs[keep_idx]] = 1
        mask = trimmed

    return mask



def _resolve_structure_warp_px(patch_size: int, warp_ratio: float = 0.05) -> float:
    return float(np.clip(float(patch_size) * float(warp_ratio), 3.0, 18.0))


def _smooth_random_flow(h: int, w: int, amplitude_px: float, rng: np.random.Generator):
    gh = max(4, min(14, h // 24))
    gw = max(4, min(14, w // 24))
    dx0 = rng.standard_normal((gh, gw)).astype(np.float32)
    dy0 = rng.standard_normal((gh, gw)).astype(np.float32)
    dx = cv2.resize(dx0, (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy0, (w, h), interpolation=cv2.INTER_CUBIC)
    sigma = max(h, w) / 32.0
    dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=sigma, sigmaY=sigma)
    dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=sigma, sigmaY=sigma)
    rms = np.sqrt(np.mean(dx * dx + dy * dy)) + 1e-6
    scale = float(amplitude_px) / float(rms)
    return dx * scale, dy * scale


def generate_structure_scramble_fill(
    img: np.ndarray,
    mask_binary: np.ndarray,
    rng: np.random.Generator,
    corruption_strength: float = 0.90,
    warp_ratio: float = 0.05,
    feather: int = 5,
) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f'Expected HxWx3 image, got {img.shape}')
    if mask_binary.shape != img.shape[:2]:
        raise ValueError('mask_binary shape must match image spatial dimensions')
    strength = float(corruption_strength)
    if not 0.0 < strength <= 1.0:
        raise ValueError('corruption_strength must be in (0, 1]')
    if warp_ratio <= 0:
        raise ValueError('warp_ratio must be > 0')

    h, w, _ = img.shape
    mask_bool = mask_binary > 0
    if not np.any(mask_bool):
        return img.copy()

    img_f = img.astype(np.float32)
    sigma = _resolve_structure_sigma(min(h, w))
    low = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    residual = img_f - low

    amp = _resolve_structure_warp_px(min(h, w), warp_ratio=warp_ratio)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    warped = []
    for _ in range(2):
        dx, dy = _smooth_random_flow(h, w, amp, rng)
        map_x = xx + dx
        map_y = yy + dy
        wr = cv2.remap(
            residual, map_x, map_y, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        warped.append(wr)

    mix0 = rng.standard_normal((max(4, h // 32), max(4, w // 32))).astype(np.float32)
    mix = cv2.resize(mix0, (w, h), interpolation=cv2.INTER_CUBIC)
    mix = cv2.GaussianBlur(mix, (0, 0), sigmaX=max(h, w) / 28.0)
    mix = 1.0 / (1.0 + np.exp(-mix))
    mix = 0.25 + 0.50 * mix
    norm = np.sqrt(mix * mix + (1.0 - mix) * (1.0 - mix)) + 1e-6
    scrambled = (mix[..., None] * warped[0] + (1.0 - mix[..., None]) * warped[1]) / norm[..., None]

    for ch in range(3):
        src_std = float(residual[..., ch][mask_bool].std())
        dst_std = float(scrambled[..., ch][mask_bool].std())
        if dst_std > 1e-6 and src_std > 1e-6:
            scale = float(np.clip(src_std / dst_std, 0.65, 1.55))
            scrambled[..., ch] *= scale

    target = np.clip(low + scrambled, 0, 255)
    corrupted = img_f * (1.0 - strength) + target * strength
    alpha = mask_bool.astype(np.float32)
    if feather > 0:
        k = _valid_blur_kernel(2 * int(feather) + 1, h, w)
        if k > 1:
            outer = cv2.GaussianBlur(alpha, (k, k), 0)
            ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            allowed = cv2.dilate(mask_bool.astype(np.uint8), ring_kernel, iterations=1) > 0
            outer[~allowed] = 0.0
            outer[mask_bool] = 1.0
            alpha = np.clip(outer, 0.0, 1.0)
    alpha = alpha[..., None]
    out = img_f * (1.0 - alpha) + corrupted * alpha
    return np.clip(out, 0, 255).astype(np.uint8)

def generate_matched_noise_fill(
    img: np.ndarray,
    mask_binary: np.ndarray,
    rng: np.random.Generator,
    local_stat_win: int = 51,
    reference_pad_ratio: float = 0.5,
    use_seamless_clone: bool = True,
    feather: int = 5,
    noise_gain: float = 0.9,
    corruption_strength: float = 0.70,
) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f'Expected HxWx3 image, got {img.shape}')
    if mask_binary.shape != img.shape[:2]:
        raise ValueError('mask_binary shape must match image spatial dimensions')
    if not 0.0 < float(corruption_strength) <= 1.0:
        raise ValueError('corruption_strength must be in (0, 1]')

    h, w, c = img.shape
    mask_bool = mask_binary > 0
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return img.copy()

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bbox_h, bbox_w = y1 - y0, x1 - x0

    pad = int(max(bbox_h, bbox_w) * reference_pad_ratio)
    ry0, ry1 = max(0, y0 - pad), min(h, y1 + pad)
    rx0, rx1 = max(0, x0 - pad), min(w, x1 + pad)
    reference_patch = img[ry0:ry1, rx0:rx1]

    _, local_std = _local_stats_maps(img, win=local_stat_win)

    unit_noise = _spectral_shaped_noise(bbox_h, bbox_w, c, reference_patch, rng)
    local_std_bbox = _robust_cap_local_std(local_std[y0:y1, x0:x1], percentile=75.0)
    structure_sigma = _resolve_structure_sigma(min(h, w))
    structure_low = cv2.GaussianBlur(
        img.astype(np.float32), (0, 0), sigmaX=structure_sigma, sigmaY=structure_sigma
    )
    noise_bbox = (
        unit_noise * local_std_bbox * float(noise_gain)
        + structure_low[y0:y1, x0:x1]
    )
    noise_bbox = np.clip(noise_bbox, 0, 255).astype(np.float32)
    original_bbox = img[y0:y1, x0:x1].astype(np.float32)
    strength = float(corruption_strength)
    corrupted_bbox = original_bbox * (1.0 - strength) + noise_bbox * strength

    canvas = img.astype(np.float32).copy()
    region_mask = mask_bool[y0:y1, x0:x1]
    canvas_bbox = canvas[y0:y1, x0:x1]
    canvas_bbox[region_mask] = corrupted_bbox[region_mask]
    canvas[y0:y1, x0:x1] = canvas_bbox
    canvas_u8 = np.clip(canvas, 0, 255).astype(np.uint8)

    result: Optional[np.ndarray] = None
    if use_seamless_clone:
        mask_u8 = (mask_bool.astype(np.uint8)) * 255
        cx = int(np.clip(round(float(xs.mean())), 1, w - 2))
        cy = int(np.clip(round(float(ys.mean())), 1, h - 2))
        try:
            result = cv2.seamlessClone(canvas_u8, img, mask_u8, (cx, cy), cv2.NORMAL_CLONE)
        except cv2.error:
            result = None

    if result is None:
        mask_f = mask_bool.astype(np.float32)
        if feather > 0:
            k = _valid_blur_kernel(2 * feather + 1, h, w)
            if k > 1:
                mask_f = cv2.GaussianBlur(mask_f, (k, k), 0)
        mask_f = np.clip(mask_f, 0.0, 1.0)[..., None]
        blended = img.astype(np.float32) * (1.0 - mask_f) + canvas.astype(np.float32) * mask_f
        result = np.clip(blended, 0, 255).astype(np.uint8)

    return result
def _list_images(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def _clear_image_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    for p in path.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            p.unlink()


def _resolve_image_path(image_dir: Path, prefix: str) -> Optional[Path]:
    for ext in ('.tif', '.tiff', '.jpg', '.jpeg', '.png'):
        p = image_dir / f'{prefix}{ext}'
        if p.exists():
            return p
    return None


def _worker_init():
    cv2.setNumThreads(1)


def _parse_size_float_map(s: str, arg_name: str) -> Dict[int, float]:
    result: Dict[int, float] = {}
    s = (s or '').strip()
    if not s:
        return result
    for chunk in s.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ':' not in chunk:
            raise ValueError(f"Invalid {arg_name} entry '{chunk}', expected 'size:value'")
        size_str, val_str = chunk.split(':', 1)
        try:
            result[int(size_str.strip())] = float(val_str.strip())
        except ValueError as e:
            raise ValueError(f"Invalid {arg_name} entry '{chunk}', expected 'size:value'") from e
    return result


def _count_by_patch_size(path: Path) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for p in _list_images(path):
        parts = p.stem.split('_')
        if len(parts) < 4:
            continue
        try:
            patch_size = int(parts[-4])
        except ValueError:
            continue
        counts[patch_size] = counts.get(patch_size, 0) + 1
    return counts


def _format_size_breakdown(counts: Dict[int, int]) -> str:
    if not counts:
        return 'total=0'
    total = sum(counts.values())
    detail = ', '.join(f'{size}px={cnt}' for size, cnt in sorted(counts.items()))
    return f'total={total} ({detail})'


def _count_by_category(path: Path) -> Dict[str, int]:
    counts = {'natural': 0, 'erased': 0, 'other': 0}
    for p in _list_images(path):
        name = p.name
        if name.startswith('natural10_'):
            counts['natural'] += 1
        elif name.startswith('erased10_'):
            counts['erased'] += 1
        else:
            counts['other'] += 1
    return counts


def _format_category_breakdown(counts: Dict[str, int]) -> str:
    total = sum(counts.values())
    if total == 0:
        return 'natural=0 (0.0%), erased=0 (0.0%)'
    nat, er = counts.get('natural', 0), counts.get('erased', 0)
    return (f'natural={nat} ({100.0 * nat / total:.1f}%), '
            f'erased={er} ({100.0 * er / total:.1f}%)')


def _make_patch_rng(
    base_seed: int,
    source_idx: int,
    av_type: int,
    patch_size: int,
    iter_idx: int,
    x: int,
    y: int,
    stream: int = 0,
) -> np.random.Generator:
    ss = np.random.SeedSequence([
        int(base_seed),
        int(source_idx),
        int(av_type),
        int(patch_size),
        int(iter_idx),
        int(x),
        int(y),
        int(stream),
    ])
    return np.random.default_rng(ss)


def _type_size_cap_reached(
    accepted: Dict[Tuple[int, int], int],
    av_type: int,
    patch_size: int,
    cap: Optional[int],
) -> bool:
    if cap is None:
        return False
    return accepted.get((int(av_type), int(patch_size)), 0) >= int(cap)


def _patch_sizes_for_av_type(av_type: int) -> List[int]:
    if av_type in (0, 1, 2):
        return [64, 128, 256, 512]
    if av_type == 3:
        return [64, 128,256]
    raise ValueError(f'Unsupported av_type: {av_type}')

def process_one_source_image(task: Dict) -> Dict:
    try:
        av_path = Path(task['av_path'])
        image_path = Path(task['image_path'])
        source_idx = task['source_idx']
        dataset_name = task['dataset_name']
        out11 = Path(task['out11'])
        out10 = Path(task['out10'])
        av_type_list: Sequence[int] = task['av_type_list']
        patch_size_arg: Sequence[int] = task['patch_size_arg']
        max_epoch_single = task['max_epoch_single']
        max_epoch_av = task['max_epoch_av']
        max_epoch_background = task['max_epoch_background']
        min_foreground_ratio = task['min_foreground_ratio']
        overlap_threshold = task['overlap_threshold']
        vessel_dilate_px = task['vessel_dilate_px']
        vessel_sample_margin = task['vessel_sample_margin']
        vessel_sigma_low = task['vessel_sigma_low']
        vessel_sigma_tex = task['vessel_sigma_tex']
        vessel_feather_px = task['vessel_feather_px']
        vessel_step_min = task['vessel_step_min']
        vessel_step_max = task['vessel_step_max']
        vessel_dilate_ratio = task['vessel_dilate_ratio']
        vessel_dilate_min_px = task['vessel_dilate_min_px']
        vessel_dilate_max_px = task['vessel_dilate_max_px']
        vessel_feather_ratio = task['vessel_feather_ratio']
        vessel_feather_min_px = task['vessel_feather_min_px']
        vessel_feather_max_px = task['vessel_feather_max_px']
        vessel_sample_margin_ratio = task['vessel_sample_margin_ratio']
        vessel_sample_margin_min_px = task['vessel_sample_margin_min_px']
        vessel_sample_margin_max_px = task['vessel_sample_margin_max_px']
        vessel_tangent_jitter_px = task['vessel_tangent_jitter_px']
        patch_size_epoch_scale: Dict[int, float] = task['patch_size_epoch_scale']
        max_patches_per_size: Dict[int, float] = task['max_patches_per_size']
        natural_background_ratio: float = task['natural_background_ratio']
        sampling_rng = np.random.default_rng(task['seed'] + source_idx)
        av = cv2.imread(str(av_path), cv2.IMREAD_COLOR)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if av is None or image_bgr is None:
            return {'rows': [], 'error': f'Failed to read pair: {image_path}, {av_path}'}
        if av.shape[:2] != image_bgr.shape[:2]:
            return {'rows': [], 'error': f'Shape mismatch {av_path.stem}: image={image_bgr.shape}, av={av.shape}'}

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        _, mask0 = creatMask(image, threshold=10)
        fov_mask = (mask0 > 0).astype(np.uint8)

        label_vessel = np.zeros(av.shape[:2], dtype=np.uint8)
        label_a = np.zeros(av.shape[:2], dtype=np.uint8)
        label_v = np.zeros(av.shape[:2], dtype=np.uint8)
        mask = np.zeros((3, av.shape[0], av.shape[1]), dtype=np.uint8)

        label_a[(av[:, :, 2] == 255) | (av[:, :, 1] == 255)] = 1
        label_a[(av[:, :, 2] == 255) & (av[:, :, 1] == 255) & (av[:, :, 0] == 255)] = 0

        label_v[(av[:, :, 1] == 255) | (av[:, :, 0] == 255)] = 1
        label_v[(av[:, :, 2] == 255) & (av[:, :, 1] == 255) & (av[:, :, 0] == 255)] = 0

        label_vessel[(av[:, :, 2] == 255) | (av[:, :, 1] == 255) | (av[:, :, 0] == 255)] = 1
        label_vessel[(av[:, :, 2] == 255) & (av[:, :, 1] == 255) & (av[:, :, 0] == 255)] = 1

        mask[0] = morphology.medial_axis(label_a).astype(np.uint8)
        mask[1] = morphology.medial_axis(label_v).astype(np.uint8)
        mask[2] = morphology.medial_axis(label_vessel).astype(np.uint8)

        rows: List[Dict] = []
        overlap_pools: Dict[Tuple[str, int], List[Tuple[int, int, int]]] = {}
        per_type_size_accepted: Dict[Tuple[int, int], int] = {}
        category_accepted: Dict[str, int] = {'erased': 0, 'natural': 0}

        def erase_vessels(
            patch_rgb: np.ndarray,
            vessel_mask2d: np.ndarray,
            erase_rng: np.random.Generator,
        ) -> np.ndarray:
            """Erase a vessel mask without touching the sampling RNG."""
            return erase_vessels_directional(
                patch_rgb,
                vessel_mask2d,
                dilate_px=vessel_dilate_px,
                sample_margin=vessel_sample_margin,
                sigma_low=vessel_sigma_low,
                sigma_tex=vessel_sigma_tex,
                feather_px=vessel_feather_px,
                step_min=vessel_step_min,
                step_max=vessel_step_max,
                dilate_ratio=vessel_dilate_ratio,
                dilate_min_px=vessel_dilate_min_px,
                dilate_max_px=vessel_dilate_max_px,
                feather_ratio=vessel_feather_ratio,
                feather_min_px=vessel_feather_min_px,
                feather_max_px=vessel_feather_max_px,
                sample_margin_ratio=vessel_sample_margin_ratio,
                sample_margin_min_px=vessel_sample_margin_min_px,
                sample_margin_max_px=vessel_sample_margin_max_px,
                tangent_jitter_px=vessel_tangent_jitter_px,
                rng=erase_rng,
            )

        requested_types = set(av_type_list)
        vessel_process_order = [t for t in (0, 1, 2) if t in requested_types]
        sampling_rng.shuffle(vessel_process_order)
        process_order = vessel_process_order + ([3] if 3 in requested_types else [])

        for av_type_i in process_order:
            natural_target_cap: Optional[int] = None
            if av_type_i == 3 and natural_background_ratio > 0.0:
                erased_so_far = category_accepted.get('erased', 0)
                natural_target_cap = int(round(
                    erased_so_far * natural_background_ratio / (1.0 - natural_background_ratio)
                ))

            if av_type_i == 2:
                max_epoch = max_epoch_av
            elif av_type_i == 3:
                max_epoch = max_epoch_background
            else:
                max_epoch = max_epoch_single
            patch_sizes = _patch_sizes_for_av_type(av_type_i)

            for patch_size_i in patch_sizes:
                scale = patch_size_epoch_scale.get(patch_size_i, 1.0)
                effective_max_epoch = max(1, int(round(max_epoch * scale)))
                cap = max_patches_per_size.get(patch_size_i)
                cap_int = int(cap) if cap is not None else None

                for iter_idx in range(effective_max_epoch):
                    if _type_size_cap_reached(
                        per_type_size_accepted, av_type_i, patch_size_i, cap_int
                    ):
                        break
                    if natural_target_cap is not None and category_accepted['natural'] >= natural_target_cap:
                        break

                    y, x, patch_size, find_patch = patch_select_for_RIP(
                        sampling_rng, patch_size_i, mask, label_a, label_v, label_vessel, av_type=av_type_i
                    )
                    if not find_patch:
                        continue

                    foreground_ratio = float(np.mean(fov_mask[y:y + patch_size, x:x + patch_size]))
                    if foreground_ratio < min_foreground_ratio:
                        continue

                    candidate = (x, y, patch_size)
                    pool_kind = 'natural' if av_type_i == 3 else 'vessel'
                    pool_key = (pool_kind, int(patch_size))
                    selected_boxes = overlap_pools.setdefault(pool_key, [])
                    if check_overlap_iou(candidate, selected_boxes, threshold=overlap_threshold):
                        continue
                    selected_boxes.append(candidate)
                    type_size_key = (av_type_i, patch_size_i)
                    per_type_size_accepted[type_size_key] = (
                        per_type_size_accepted.get(type_size_key, 0) + 1
                    )
                    category_key = 'natural' if av_type_i == 3 else 'erased'
                    category_accepted[category_key] = category_accepted.get(category_key, 0) + 1

                    patch_ve = label_vessel[y:y + patch_size, x:x + patch_size]
                    patch_a = label_a[y:y + patch_size, x:x + patch_size]
                    patch_v = label_v[y:y + patch_size, x:x + patch_size]
                    patch_img = image[y:y + patch_size, x:x + patch_size, :]

                    base = f'{dataset_name}_{source_idx}_{av_type_i}_{patch_size}_{iter_idx}_{x}_{y}.png'

                    def patch_rng(stream: int) -> np.random.Generator:
                        return _make_patch_rng(
                            task['seed'], source_idx, av_type_i, patch_size,
                            iter_idx, x, y, stream=stream,
                        )

                    if av_type_i == 0:
                        patch_nve = erase_vessels(patch_img, patch_ve, patch_rng(0))
                        Image.fromarray(patch_img).save(out11 / f'raw11_a_{base}')
                        Image.fromarray(patch_nve).save(out10 / f'erased10_a_{base}')

                    elif av_type_i == 1:
                        patch_nve = erase_vessels(patch_img, patch_ve, patch_rng(0))
                        Image.fromarray(patch_img).save(out11 / f'raw11_v_{base}')
                        Image.fromarray(patch_nve).save(out10 / f'erased10_v_{base}')

                    elif av_type_i == 2:
                        choice = int(sampling_rng.integers(0, 3))
                        if choice == 0:
                            img11 = erase_vessels(patch_img, patch_v, patch_rng(1))
                            tag = 'partial11_keepA'
                        elif choice == 1:
                            img11 = erase_vessels(patch_img, patch_a, patch_rng(1))
                            tag = 'partial11_keepV'
                        else:
                            img11 = patch_img
                            tag = 'raw11_keepAV'
                        patch_nve = erase_vessels(patch_img, patch_ve, patch_rng(2))
                        Image.fromarray(img11).save(out11 / f'{tag}_{base}')
                        Image.fromarray(patch_nve).save(out10 / f'erased10_av_{base}')

                    elif av_type_i == 3:
                        Image.fromarray(patch_img).save(out10 / f'natural10_b_{base}')

                    rows.append({
                        'source_idx': source_idx,
                        'av_path': str(av_path),
                        'image_path': str(image_path),
                        'patch_size': int(patch_size),
                        'x': int(x),
                        'y': int(y),
                        'av_type': int(av_type_i),
                    })

        return {'rows': rows, 'error': None}
    except Exception:
        return {'rows': [], 'error': f'{task.get("av_path")}: {traceback.format_exc()}'}

def process_one_noise_overlay(task: Dict) -> Dict:
    try:
        img_path = Path(task['img_path'])
        save_path = Path(task['save_path'])
        rng = np.random.default_rng(task['seed'])

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            return {'row': None, 'error': f'Failed to read source image: {img_path}'}

        h, w = img.shape[:2]

        rgb_for_fov = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, fov_mask = creatMask(rgb_for_fov, threshold=10)
        base_safe_radius = _resolve_adaptive_px(
            min(h, w), fixed_px=0, ratio=0.015, min_px=2, max_px=8
        )
        warp_safe_radius = int(np.ceil(_resolve_structure_warp_px(
            min(h, w), warp_ratio=task['structure_warp_ratio']
        )))
        safe_radius = max(base_safe_radius, warp_safe_radius + 2)
        if safe_radius > 0:
            k = 2 * safe_radius + 1
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            safe_fov = cv2.erode((fov_mask > 0).astype(np.uint8), ker, iterations=1)
        else:
            safe_fov = (fov_mask > 0).astype(np.uint8)
        if safe_fov.sum() < max(16, int(0.20 * (fov_mask > 0).sum())):
            safe_fov = (fov_mask > 0).astype(np.uint8)

        noise_mask = _sample_irregular_overlay_mask(
            h, w, task['area_min'], task['area_max'], rng, valid_mask=safe_fov
        )
        valid_fraction = float(noise_mask.sum() / max(int(safe_fov.sum()), 1))
        ys_mask, xs_mask = np.where(noise_mask > 0)
        y0, y1 = int(ys_mask.min()), int(ys_mask.max()) + 1
        x0, x1 = int(xs_mask.min()), int(xs_mask.max()) + 1
        rect_h, rect_w = y1 - y0, x1 - x0

        result = generate_structure_scramble_fill(
            img, noise_mask, rng,
            corruption_strength=task['corruption_strength'],
            warp_ratio=task['structure_warp_ratio'],
            feather=task['feather'],
        )

        ok = cv2.imwrite(str(save_path), result)
        if not ok:
            return {'row': None, 'error': f'Failed to write: {save_path}'}

        src_mean, src_std = cv2.meanStdDev(img)
        out_mean, out_std = cv2.meanStdDev(result)
        row = {
            'source_path': str(img_path),
            'noise_path': str(save_path),
            'area_fraction_target': float(noise_mask.mean()),
            'mask_kind': 'irregular_structure_scramble',
            'area_fraction_of_valid_retina': valid_fraction,
            'corruption_strength': float(task['corruption_strength']),
            'structure_warp_ratio': float(task['structure_warp_ratio']),
            'rect_y0': int(y0), 'rect_x0': int(x0),
            'rect_h': int(rect_h), 'rect_w': int(rect_w),
            'source_mean_b': float(src_mean[0, 0]),
            'source_mean_g': float(src_mean[1, 0]),
            'source_mean_r': float(src_mean[2, 0]),
            'source_std_b': float(src_std[0, 0]),
            'source_std_g': float(src_std[1, 0]),
            'source_std_r': float(src_std[2, 0]),
            'noise_mean_b': float(out_mean[0, 0]),
            'noise_mean_g': float(out_mean[1, 0]),
            'noise_mean_r': float(out_mean[2, 0]),
            'noise_std_b': float(out_std[0, 0]),
            'noise_std_g': float(out_std[1, 0]),
            'noise_std_r': float(out_std[2, 0]),
        }
        return {'row': row, 'error': None}
    except Exception:  # noqa: BLE001
        return {'row': None, 'error': f'{task.get("img_path")}: {traceback.format_exc()}'}
def build_source_tasks(args, out11: Path, out10: Path,
                        patch_size_epoch_scale: Dict[int, float],
                        max_patches_per_size: Dict[int, float],
                        natural_background_ratio: float) -> List[Dict]:
    dataset_path = Path(args.dataset_path)
    split = args.train_or_test
    dataset_name = dataset_path.name

    av_dir = dataset_path / split / 'av'
    image_dir = dataset_path / split / 'images'
    if not av_dir.exists():
        raise FileNotFoundError(f'AV directory not found: {av_dir}')
    if not image_dir.exists():
        raise FileNotFoundError(f'Image directory not found: {image_dir}')

    av_paths = sorted([p for p in av_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not av_paths:
        raise RuntimeError(f'No AV label images found in {av_dir}')

    tasks = []
    skipped = 0
    for source_idx, av_path in enumerate(av_paths):
        image_path = _resolve_image_path(image_dir, av_path.stem)
        if image_path is None:
            skipped += 1
            continue
        tasks.append({
            'av_path': str(av_path),
            'image_path': str(image_path),
            'source_idx': source_idx,
            'dataset_name': dataset_name,
            'out11': str(out11),
            'out10': str(out10),
            'av_type_list': args.av_type,
            'patch_size_arg': args.patch_size,
            'max_epoch_single': args.max_epoch_single,
            'max_epoch_av': args.max_epoch_av,
            'max_epoch_background': args.max_epoch_background,
            'min_foreground_ratio': args.min_foreground_ratio,
            'overlap_threshold': args.overlap_threshold,
            'vessel_dilate_px': args.vessel_dilate_px,
            'vessel_sample_margin': args.vessel_sample_margin,
            'vessel_sigma_low': args.vessel_sigma_low,
            'vessel_sigma_tex': args.vessel_sigma_tex,
            'vessel_feather_px': args.vessel_feather_px,
            'vessel_step_min': args.vessel_step_min,
            'vessel_step_max': args.vessel_step_max,
            'vessel_dilate_ratio': args.vessel_dilate_ratio,
            'vessel_dilate_min_px': args.vessel_dilate_min_px,
            'vessel_dilate_max_px': args.vessel_dilate_max_px,
            'vessel_feather_ratio': args.vessel_feather_ratio,
            'vessel_feather_min_px': args.vessel_feather_min_px,
            'vessel_feather_max_px': args.vessel_feather_max_px,
            'vessel_sample_margin_ratio': args.vessel_sample_margin_ratio,
            'vessel_sample_margin_min_px': args.vessel_sample_margin_min_px,
            'vessel_sample_margin_max_px': args.vessel_sample_margin_max_px,
            'vessel_tangent_jitter_px': args.vessel_tangent_jitter_px,
            'seed': args.seed,
            'patch_size_epoch_scale': patch_size_epoch_scale,
            'max_patches_per_size': max_patches_per_size,
            'natural_background_ratio': natural_background_ratio,
        })
    if skipped:
        print(f'[WARN] Skipped {skipped} AV files with no matching fundus image.')
    return tasks


def run_stage1(args, out11: Path, out10: Path,
                patch_size_epoch_scale: Dict[int, float],
                max_patches_per_size: Dict[int, float],
                natural_background_ratio: float) -> pd.DataFrame:
    tasks = build_source_tasks(args, out11, out10, patch_size_epoch_scale, max_patches_per_size,
                                natural_background_ratio)
    print(f'[Stage 1] {len(tasks)} source images, {args.num_workers} workers.')

    all_rows: List[Dict] = []
    errors: List[str] = []

    with mp.Pool(processes=args.num_workers, initializer=_worker_init) as pool:
        for result in tqdm(pool.imap_unordered(process_one_source_image, tasks),
                            total=len(tasks), desc='Stage1: 11/10 patches'):
            if result['error']:
                errors.append(result['error'])
            all_rows.extend(result['rows'])

    if errors:
        print(f'\n[Stage 1] {len(errors)} source images reported errors/warnings, e.g.:')
        for e in errors[:10]:
            print(f'  - {e.splitlines()[0] if isinstance(e, str) else e}')

    return pd.DataFrame(all_rows)


def run_stage2(args, out10: Path, out00: Path) -> pd.DataFrame:
    img_files = _list_images(out10)
    if not img_files:
        raise RuntimeError(f'No images found in {out10}; cannot generate images00.')

    tasks = []
    for i, img_path in enumerate(img_files):
        save_path = out00 / f'noise_{img_path.name}'
        tasks.append({
            'img_path': str(img_path),
            'save_path': str(save_path),
            'seed': args.seed + 1_000_000 + i,
            'area_min': args.noise_area_min,
            'area_max': args.noise_area_max,
            'local_stat_win': args.noise_local_stat_win,
            'reference_pad_ratio': args.noise_reference_pad_ratio,
            'use_seamless_clone': not args.noise_no_seamless_clone,
            'feather': args.noise_feather,
            'noise_gain': args.noise_gain,
            'corruption_strength': args.noise_corruption_strength,
            'structure_warp_ratio': args.structure_warp_ratio,
        })

    print(f'[Stage 2] {len(tasks)} images10 -> images00, {args.num_workers} workers.')

    rows: List[Dict] = []
    errors: List[str] = []
    with mp.Pool(processes=args.num_workers, initializer=_worker_init) as pool:
        for result in tqdm(pool.imap_unordered(process_one_noise_overlay, tasks),
                            total=len(tasks), desc='Stage2: images00'):
            if result['error']:
                errors.append(result['error'])
            elif result['row'] is not None:
                rows.append(result['row'])

    if errors:
        print(f'\n[Stage 2] {len(errors)} files reported errors, e.g.:')
        for e in errors[:10]:
            print(f'  - {e.splitlines()[0] if isinstance(e, str) else e}')

    out_files = _list_images(out00)
    if len(out_files) != len(img_files):
        raise RuntimeError(
            f'Balance check failed: images10={len(img_files)}, images00={len(out_files)}'
        )

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='../data/STU',
                         help='dataset root containing <split>/images and <split>/av')
    parser.add_argument('--train_or_test', type=str, choices=['training', 'test'], default='test')
    parser.add_argument('--av_type', nargs='+', type=int, default=[0, 1, 2, 3], choices=[0, 1, 2, 3])
    parser.add_argument('--patch_size', nargs='+', type=int, default=[64, 128, 256, 512],
                         help='legacy compatibility argument; sampling sizes are fixed by av_type: '
                              '0/1/2 -> 64,128,256,512 and 3 -> 64,128')
    parser.add_argument('--output', type=str, default='./', help='output root')
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--max_epoch_single', type=int, default=200)
    parser.add_argument('--max_epoch_av', type=int, default=200)
    parser.add_argument('--max_epoch_background', type=int, default=200)
    parser.add_argument('--min_foreground_ratio', type=float, default=0.7)
    parser.add_argument('--overlap_threshold', type=float, default=0.3,
                         help='maximum allowed IoU within each same-size overlap pool; av_type '
                              '0/1/2 share a vessel pool and av_type 3 uses a separate natural pool; '
                              'IoU > threshold is rejected')
    parser.add_argument('--patch_size_epoch_scale', type=str, default='64:0.4')
    parser.add_argument('--max_patches_per_size', type=str, default='')
    parser.add_argument('--natural_background_ratio', type=float, default=0.5)
    parser.add_argument('--vessel_dilate_px', type=int, default=0)
    parser.add_argument('--vessel_dilate_ratio', type=float, default=0.02)
    parser.add_argument('--vessel_dilate_min_px', type=int, default=3)
    parser.add_argument('--vessel_dilate_max_px', type=int, default=12)
    parser.add_argument('--vessel_sample_margin', type=int, default=0)
    parser.add_argument('--vessel_sample_margin_ratio', type=float, default=0.015)
    parser.add_argument('--vessel_sample_margin_min_px', type=int, default=4)
    parser.add_argument('--vessel_sample_margin_max_px', type=int, default=10)
    parser.add_argument('--vessel_sigma_low', type=float, default=5.0)
    parser.add_argument('--vessel_sigma_tex', type=float, default=2.0)
    parser.add_argument('--vessel_feather_px', type=float, default=0)
    parser.add_argument('--vessel_feather_ratio', type=float, default=0.008)
    parser.add_argument('--vessel_feather_min_px', type=int, default=2)
    parser.add_argument('--vessel_feather_max_px', type=int, default=6)
    parser.add_argument('--vessel_tangent_jitter_px', type=int, default=2)
    parser.add_argument('--vessel_step_min', type=int, default=2)
    parser.add_argument('--vessel_step_max', type=int, default=32)
    parser.add_argument('--noise_area_min', type=float, default=0.40,
                         help='(images00 only) min local irregular area fraction of images10 covered by noise')
    parser.add_argument('--noise_area_max', type=float, default=0.65,
                         help='(images00 only) max local irregular area fraction of images10 covered by noise')
    parser.add_argument('--noise_local_stat_win', type=int, default=51,
                         help='window size (px) for the local mean/std field used to color the noise')
    parser.add_argument('--noise_reference_pad_ratio', type=float, default=0.5,
                         help='extra context (relative to erased-region size) sampled for spectral matching')
    parser.add_argument('--noise_no_seamless_clone', action='store_true',
                         help='use feathered alpha blending instead of Poisson seamlessClone at the edge')
    parser.add_argument('--noise_feather', type=int, default=5,
                         help='feather radius (px) used for large 00 irregular-mask blending')
    parser.add_argument('--noise_gain', type=float, default=0.90,
                         help='00 texture-noise contrast multiplier; default 0.90')
    parser.add_argument('--noise_corruption_strength', type=float, default=0.90,
                         help='V6 local structure-scramble strength; default 0.90')
    parser.add_argument('--structure_warp_ratio', type=float, default=0.05,
                         help='V6 high-frequency retinal-structure warp amplitude relative to patch size; default 0.05')

    parser.add_argument('--num_workers', type=int, default=64, help='multiprocessing pool size')
    parser.add_argument('--clear_output', action='store_true',
                         help='clear images11/images10/images00 before generating')
    parser.add_argument('--skip_stage1', action='store_true',
                         help='skip 11/10 generation and only (re)generate images00 from existing images10')
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.patch_size or any(p <= 0 for p in args.patch_size):
        raise ValueError('--patch_size must contain positive integers')
    if not 0.0 <= args.min_foreground_ratio <= 1.0:
        raise ValueError('--min_foreground_ratio must be in [0, 1]')
    if not 0.0 <= args.overlap_threshold <= 1.0:
        raise ValueError('--overlap_threshold must be in [0, 1]')
    if not 0.0 < args.noise_area_min <= args.noise_area_max <= 1.0:
        raise ValueError('Require 0 < --noise_area_min <= --noise_area_max <= 1')
    if args.noise_feather < 0:
        raise ValueError('--noise_feather must be >= 0')
    if args.noise_gain <= 0:
        raise ValueError('--noise_gain must be > 0')
    if not 0.0 < args.noise_corruption_strength <= 1.0:
        raise ValueError('--noise_corruption_strength must be in (0, 1]')
    if args.structure_warp_ratio <= 0:
        raise ValueError('--structure_warp_ratio must be > 0')
    if args.num_workers <= 0:
        raise ValueError('--num_workers must be > 0')

    if args.vessel_dilate_px < 0:
        raise ValueError('--vessel_dilate_px must be >= 0')
    if args.vessel_sample_margin < 0:
        raise ValueError('--vessel_sample_margin must be >= 0')
    if args.vessel_sigma_low <= 0 or args.vessel_sigma_tex <= 0 or args.vessel_feather_px < 0:
        raise ValueError('--vessel_sigma_low / --vessel_sigma_tex must be > 0 and --vessel_feather_px >= 0')
    if args.vessel_dilate_ratio <= 0 or args.vessel_feather_ratio <= 0 or args.vessel_sample_margin_ratio <= 0:
        raise ValueError('adaptive vessel ratios must be > 0')
    if args.vessel_tangent_jitter_px < 0:
        raise ValueError('--vessel_tangent_jitter_px must be >= 0')
    if args.vessel_step_min < 1 or args.vessel_step_max < args.vessel_step_min:
        raise ValueError('Require 1 <= --vessel_step_min <= --vessel_step_max')

    patch_size_epoch_scale = _parse_size_float_map(args.patch_size_epoch_scale, '--patch_size_epoch_scale')
    for size, scale in patch_size_epoch_scale.items():
        if scale <= 0:
            raise ValueError(f'--patch_size_epoch_scale for size {size} must be > 0 (got {scale})')
    max_patches_per_size = _parse_size_float_map(args.max_patches_per_size, '--max_patches_per_size')
    for size, cap in max_patches_per_size.items():
        if cap < 0:
            raise ValueError(f'--max_patches_per_size for size {size} must be >= 0 (got {cap})')
    if not 0.0 <= args.natural_background_ratio < 1.0:
        raise ValueError('--natural_background_ratio must be in [0, 1)')

    split_output = Path(args.output) / args.train_or_test
    out11 = split_output / 'images11'
    out10 = split_output / 'images10'
    out00 = split_output / 'images00'
    for d in (out11, out10, out00):
        d.mkdir(parents=True, exist_ok=True)

    if args.clear_output:
        _clear_image_dir(out11)
        _clear_image_dir(out10)
        _clear_image_dir(out00)

    dataset_name = Path(args.dataset_path).name

    if not args.skip_stage1:
        patch_df = run_stage1(args, out11, out10, patch_size_epoch_scale, max_patches_per_size,
                               args.natural_background_ratio)
        patch_path = split_output / f'{dataset_name}_{args.train_or_test}_patch_locations.csv'
        patch_df.to_csv(patch_path, index=False)
        print(f'Patch locations: {patch_path}')

    _clear_image_dir(out00)
    noise_df = run_stage2(args, out10, out00)
    noise_manifest_path = split_output / 'images00_manifest.csv'
    noise_df.to_csv(noise_manifest_path, index=False)

    n11 = len(_list_images(out11))
    n10 = len(_list_images(out10))
    n00 = len(_list_images(out00))

    print('\nFinal counts (by patch size, parsed from saved filenames):')
    print(f'  images11  ([1,1]): {_format_size_breakdown(_count_by_patch_size(out11))}')
    print(f'  images10  ([1,0]): {_format_size_breakdown(_count_by_patch_size(out10))}')
    print(f'  images00  ([0,0]): {_format_size_breakdown(_count_by_patch_size(out00))}')
    print(f'  images10 natural(av_type=3) vs erased(av_type=0/1/2): '
          f'{_format_category_breakdown(_count_by_category(out10))}')
    if n10 != n00:
        raise RuntimeError(f'Balance check failed: images10={n10}, images00={n00}')
    if not args.skip_stage1 and n11 != n10:
        print(f'  [WARN] images11 ({n11}) != images10 ({n10}); this simple pipeline does not '
              f'force 11/10 balance (unlike the earlier balanced-selection version).')
    print(f'images00 manifest: {noise_manifest_path}')


if __name__ == '__main__':
    main()



