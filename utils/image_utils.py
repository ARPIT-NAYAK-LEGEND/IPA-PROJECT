import base64
import io
import os
from typing import Tuple, Union, Optional
import cv2
import numpy as np
from PIL import Image


def validate_image(img: np.ndarray, name: str = "Image") -> None:
    if not isinstance(img, np.ndarray):
        raise ValueError(f"{name} must be a numpy ndarray, got {type(img)}")
    if img.size == 0:
        raise ValueError(f"{name} array is empty.")
    if len(img.shape) not in (2, 3):
        raise ValueError(f"{name} must have 2 or 3 dimensions, got shape {img.shape}.")
    if len(img.shape) == 3 and img.shape[2] not in (1, 3, 4):
        raise ValueError(f"{name} has invalid channel count {img.shape[2]}. Expected 1, 3, or 4.")
    if img.shape[0] < 2 or img.shape[1] < 2:
        raise ValueError(f"{name} resolution ({img.shape[1]}x{img.shape[0]}) is too small. Minimum is 2x2.")


def to_rgb(img: np.ndarray) -> np.ndarray:
    validate_image(img, "Input")

    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            img = (img * 255.0 if img.max() <= 1.0 else img).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if len(img.shape) == 3:
        if img.shape[2] == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 3:
            return img.copy()
        if img.shape[2] == 4:
            rgb = img[:, :, :3].astype(np.float32)
            alpha = (img[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
            white = np.ones_like(rgb) * 255.0
            return np.clip(rgb * alpha + white * (1.0 - alpha), 0, 255).astype(np.uint8)

    raise ValueError(f"Unsupported image shape: {img.shape}")


def to_gray(img: np.ndarray) -> np.ndarray:
    validate_image(img, "Input")

    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating) and img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    if len(img.shape) == 2:
        return img.copy()
    if len(img.shape) == 3:
        if img.shape[2] == 1:
            return img[:, :, 0].copy()
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if img.shape[2] == 4:
            return cv2.cvtColor(to_rgb(img), cv2.COLOR_RGB2GRAY)

    raise ValueError(f"Cannot convert shape {img.shape} to grayscale.")


def to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(to_rgb(img), cv2.COLOR_RGB2BGR)


def load_image(
    source: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray],
    target_mode: str = "RGB"
) -> np.ndarray:
    if isinstance(source, np.ndarray):
        img_array = source.copy()
    elif isinstance(source, Image.Image):
        if source.mode == "RGBA":
            bg = Image.new("RGB", source.size, (255, 255, 255))
            bg.paste(source, mask=source.split()[3])
            img_array = np.array(bg)
        elif source.mode == "L":
            img_array = np.array(source)
        else:
            img_array = np.array(source.convert("RGB"))
    elif isinstance(source, (bytes, io.BytesIO)):
        raw = source.getvalue() if isinstance(source, io.BytesIO) else source
        if not raw:
            raise ValueError("Provided byte buffer is empty.")
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError("Failed to decode image from buffer.")
        if len(decoded.shape) == 3 and decoded.shape[2] == 4:
            b, g, r, a = cv2.split(decoded)
            alpha = a.astype(float) / 255.0
            bgr = cv2.merge([b, g, r]).astype(float)
            white = np.ones_like(bgr) * 255.0
            out = cv2.convertScaleAbs(bgr * alpha[:, :, None] + white * (1.0 - alpha[:, :, None]))
            img_array = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        elif len(decoded.shape) == 3 and decoded.shape[2] == 3:
            img_array = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        else:
            img_array = decoded
    elif isinstance(source, str):
        if not os.path.exists(source):
            raise ValueError(f"Image not found: {source}")
        decoded = cv2.imread(source, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError(f"Could not read image file: {source}")
        if len(decoded.shape) == 3 and decoded.shape[2] == 4:
            b, g, r, a = cv2.split(decoded)
            alpha = a.astype(float) / 255.0
            bgr = cv2.merge([b, g, r]).astype(float)
            white = np.ones_like(bgr) * 255.0
            out = cv2.convertScaleAbs(bgr * alpha[:, :, None] + white * (1.0 - alpha[:, :, None]))
            img_array = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        elif len(decoded.shape) == 3 and decoded.shape[2] == 3:
            img_array = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        else:
            img_array = decoded
    else:
        raise TypeError(f"Unsupported image source: {type(source)}")

    if img_array is None or img_array.size == 0:
        raise ValueError("Decoded image is empty.")

    validate_image(img_array, "Loaded image")

    mode = target_mode.upper()
    if mode == "RGB":
        return to_rgb(img_array)
    if mode == "BGR":
        return to_bgr(img_array)
    if mode == "GRAY":
        return to_gray(img_array)
    return img_array


def normalize_image(
    image: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 1.0,
    dtype: type = np.float32
) -> np.ndarray:
    validate_image(image, "Input")
    img_f = image.astype(np.float64)
    c_min = float(np.min(img_f))
    c_max = float(np.max(img_f))

    if c_max == c_min:
        return np.full_like(image, min_val, dtype=dtype)

    scaled = (img_f - c_min) / (c_max - c_min)
    return (scaled * (max_val - min_val) + min_val).astype(dtype)


def denormalize_image(
    image: np.ndarray,
    current_range: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    r_min, r_max = current_range
    if r_max == r_min:
        return np.zeros_like(image, dtype=np.uint8)

    scaled = (image.astype(np.float64) - r_min) / (r_max - r_min)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def calculate_aspect_ratio(img: np.ndarray) -> float:
    h, w = img.shape[:2]
    return float(w) / float(h) if h > 0 else 1.0


def resize_image(
    img: np.ndarray,
    target_size: Tuple[int, int],
    preserve_aspect_ratio: bool = True,
    pad_color: Union[int, Tuple[int, int, int]] = 0
) -> np.ndarray:
    validate_image(img, "Image")
    tw, th = target_size
    if tw <= 0 or th <= 0:
        raise ValueError(f"Target dimensions must be positive integers, got ({tw}, {th}).")

    ih, iw = img.shape[:2]
    if (iw, ih) == (tw, th):
        return img.copy()

    if not preserve_aspect_ratio:
        interp = cv2.INTER_AREA if (tw < iw or th < ih) else cv2.INTER_CUBIC
        return cv2.resize(img, (tw, th), interpolation=interp)

    scale = min(tw / iw, th / ih)
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    scaled = cv2.resize(img, (new_w, new_h), interpolation=interp)

    pad_top = (th - new_h) // 2
    pad_bottom = th - new_h - pad_top
    pad_left = (tw - new_w) // 2
    pad_right = tw - new_w - pad_left

    val = list(pad_color) if (len(img.shape) == 3 and isinstance(pad_color, (tuple, list))) else pad_color
    return cv2.copyMakeBorder(
        scaled, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=val
    )


def ensure_same_dimensions(
    img1: np.ndarray,
    img2: np.ndarray,
    mode: str = "resize",
    target_reference: str = "first",
    preserve_aspect_ratio: bool = True
) -> Tuple[np.ndarray, np.ndarray, dict]:
    validate_image(img1, "Image A")
    validate_image(img2, "Image B")

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    meta = {
        "original_shape_a": (h1, w1),
        "original_shape_b": (h2, w2),
        "mode": mode,
        "adjusted": False,
        "final_shape": (h1, w1),
    }

    if (h1, w1) == (h2, w2):
        return img1.copy(), img2.copy(), meta

    meta["adjusted"] = True

    if target_reference == "first":
        tw, th = w1, h1
    elif target_reference == "max":
        tw, th = max(w1, w2), max(h1, h2)
    elif target_reference == "min":
        tw, th = min(w1, w2), min(h1, h2)
    else:
        tw, th = w1, h1

    meta["final_shape"] = (th, tw)
    use_pad = (mode == "pad") or preserve_aspect_ratio

    out1 = resize_image(img1, (tw, th), preserve_aspect_ratio=use_pad) if (w1, h1) != (tw, th) else img1.copy()
    out2 = resize_image(img2, (tw, th), preserve_aspect_ratio=use_pad) if (w2, h2) != (tw, th) else img2.copy()

    return out1, out2, meta


def to_pil(img: np.ndarray) -> Image.Image:
    if len(img.shape) == 2:
        return Image.fromarray(img.astype(np.uint8), mode="L")
    return Image.fromarray(to_rgb(img), mode="RGB")


def to_bytes(img: np.ndarray, format: str = "PNG", quality: int = 95) -> bytes:
    pil_img = to_pil(img)
    buf = io.BytesIO()
    if format.upper() in ("JPG", "JPEG"):
        pil_img.save(buf, format="JPEG", quality=quality)
    else:
        pil_img.save(buf, format="PNG")
    return buf.getvalue()


def encode_image_to_base64(img: np.ndarray, format: str = "PNG") -> str:
    raw_bytes = to_bytes(img, format=format)
    b64_str = base64.b64encode(raw_bytes).decode("utf-8")
    mime = "image/png" if format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64_str}"
