"""Pilot CT preprocessing. Heuristic masks are not clinical segmentations."""
import numpy as np
import pydicom
from scipy import ndimage as ndi
from scipy.ndimage import zoom

WINDOW_LOW = -1000.0
WINDOW_HIGH = 400.0
TARGET_SPACING_MM = 1.0
PATCH_SIZE = 64
MIN_LUNG_FRACTION = 0.80

def signed_padding_value(value, ds):
    value = int(value)
    bits = int(ds.BitsAllocated)
    if int(ds.PixelRepresentation) == 1 and value >= 2 ** (bits - 1):
        value -= 2 ** bits
    return value

def read_ct_slice(path):
    ds = pydicom.dcmread(str(path))
    pixels = ds.pixel_array
    padding_mask = np.zeros(pixels.shape, dtype=bool)

    if "PixelPaddingValue" in ds:
        padding = signed_padding_value(ds.PixelPaddingValue, ds)

        if "PixelPaddingRangeLimit" in ds:
            limit = signed_padding_value(ds.PixelPaddingRangeLimit, ds)
            low, high = sorted((padding, limit))
            padding_mask = (pixels >= low) & (pixels <= high)
        else:
            padding_mask = pixels == padding

    image = (
        pixels.astype(np.float32) * float(ds.RescaleSlope)
        + float(ds.RescaleIntercept)
    )
    image[padding_mask] = np.nan
    return image, padding_mask, ds

def normalize_ct(image):
    filled = np.where(np.isfinite(image), image, WINDOW_LOW)
    clipped = np.clip(filled, WINDOW_LOW, WINDOW_HIGH)
    return (
        2.0 * (clipped - WINDOW_LOW) / (WINDOW_HIGH - WINDOW_LOW) - 1.0
    ).astype(np.float32)

def resample_ct_slice(image, padding_mask, ds):
    spacing = np.array(ds.PixelSpacing, dtype=float)
    factors = spacing / TARGET_SPACING_MM
    filled = np.where(np.isfinite(image), image, WINDOW_LOW)

    sigma = np.maximum((1.0 / factors - 1.0) / 2.0, 0.0)
    smoothed = ndi.gaussian_filter(filled, sigma=sigma)

    resampled = zoom(
        smoothed, factors, order=1, prefilter=False
    ).astype(np.float32)

    safe_valid = ndi.binary_erosion(
        ~padding_mask, iterations=1, border_value=0
    )
    valid_mask = zoom(
        safe_valid.astype(np.uint8),
        factors, order=0, prefilter=False
    ).astype(bool)

    resampled[~valid_mask] = np.nan
    return resampled, valid_mask

def lung_region_mask(image, valid_mask):
    tissue = valid_mask & np.isfinite(image) & (image > -500)
    labels, count = ndi.label(tissue)

    if count == 0:
        return np.zeros(image.shape, dtype=bool)

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    body = ndi.binary_fill_holes(labels == sizes.argmax())

    candidates = body & valid_mask & (image < -500)
    labels, count = ndi.label(candidates)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    keep = np.flatnonzero(sizes >= 200)
    mask = np.isin(labels, keep)
    return ndi.binary_fill_holes(mask)

def keep_two_largest_regions(mask):
    labels, count = ndi.label(mask)

    if count == 0:
        return np.zeros(mask.shape, dtype=bool)

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    region_ids = np.flatnonzero(sizes > 0)
    largest = region_ids[np.argsort(sizes[region_ids])[-2:]]

    return np.isin(labels, largest)

def extract_slice_patches(path, rng, count=8, max_attempts=5000):
    img, padding, metadata = read_ct_slice(path)
    sampled, valid = resample_ct_slice(img, padding, metadata)
    mask = keep_two_largest_regions(lung_region_mask(sampled, valid))

    height, width = sampled.shape
    patches, coordinates = [], []
    used = set()

    if min(height, width) < PATCH_SIZE:
        return patches, coordinates

    for _ in range(max_attempts):
        row = int(rng.integers(0, height - PATCH_SIZE + 1))
        col = int(rng.integers(0, width - PATCH_SIZE + 1))

        if (row, col) in used:
            continue

        region = np.s_[row:row + PATCH_SIZE, col:col + PATCH_SIZE]

        if not valid[region].all():
            continue
        if mask[region].mean() < MIN_LUNG_FRACTION:
            continue
        if not np.isfinite(sampled[region]).all():
            continue

        patches.append(normalize_ct(sampled[region]))
        coordinates.append((row, col))
        used.add((row, col))

        if len(patches) >= count:
            break

    return patches, coordinates
