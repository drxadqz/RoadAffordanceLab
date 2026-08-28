from __future__ import annotations

# 图像预处理阅读提示：正式训练使用 RandomResizedCrop(224)、水平翻转和轻量
# 仿射；正式验证使用 RSPNet 风格“短边缩放到约 256，再中心裁剪 224”。最后
# 统一 ToTensor + ImageNet mean/std。下面其他增强类用于历史消融，正式 YAML
# 的概率为 0。续训时不要改变随机增强配置，否则同一 RNG 状态会生成不同图像。

import io
import random
from collections.abc import Sequence

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def _normalize_output_hw(image_size: int | Sequence[int]) -> tuple[int, int]:
    """Return ``(height, width)`` without changing the legacy scalar path."""

    if isinstance(image_size, bool):
        raise TypeError("image_size must be an integer or a two-item sequence")
    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        return int(image_size), int(image_size)
    if isinstance(image_size, Sequence) and not isinstance(image_size, (str, bytes)):
        values = list(image_size)
        if len(values) != 2:
            raise ValueError("rectangular image_size must contain (height, width)")
        height, width = int(values[0]), int(values[1])
        if height <= 0 or width <= 0:
            raise ValueError("image height and width must be positive")
        return height, width
    raise TypeError("image_size must be an integer or a two-item sequence")


def build_transforms(
    image_size: int | Sequence[int],
    train: bool = True,
    aug_cfg: dict | None = None,
):
    """根据 YAML 构造训练或验证预处理流水线。

    ``image_size`` 是输出尺寸（正式实验 224）；``train=True`` 才允许随机增强；
    ``aug_cfg`` 对应 YAML 的增强/resize 参数。返回 Compose，最终输出归一化的
    ``[3,H,W]`` 张量。
    """
    aug_cfg = aug_cfg or {}
    ops = []
    output_hw = _normalize_output_hw(image_size)
    legacy_square = isinstance(image_size, int) and not isinstance(image_size, bool)
    resize_mode = str(aug_cfg.get("resize_mode", "stretch")).lower()
    # 训练几何增强会消耗 RNG，精确续训必须先恢复 RNG 再取下一 batch。
    if train and bool(aug_cfg.get("random_resized_crop", False)):
        scale = tuple(aug_cfg.get("crop_scale", [0.75, 1.0]))
        ratio = tuple(aug_cfg.get("crop_ratio", [0.9, 1.1]))
        crop_size = image_size if legacy_square else output_hw
        ops.append(transforms.RandomResizedCrop(crop_size, scale=scale, ratio=ratio))
    # 验证几何保持长宽比且不含随机性。
    elif resize_mode in {
        "resize_center_crop",
        "short_side_center_crop",
        "rspnet_center_crop",
    }:
        # Match the public RSPNet validation geometry: resize the shorter
        # image edge to 1.14 x the requested square input (256 for 224), then
        # take a deterministic center crop.  Unlike a direct HxW resize this
        # preserves aspect ratio, which is important for the directional
        # evidence filters used by DREL-RT.
        resize_scale = float(aug_cfg.get("resize_scale", 1.14))
        if resize_scale < 1.0:
            raise ValueError("resize_center_crop resize_scale must be >= 1")
        if output_hw[0] != output_hw[1]:
            raise ValueError("resize_center_crop currently requires square output")
        short_edge = max(output_hw[0], int(round(output_hw[0] * resize_scale)))
        ops.append(transforms.Resize(short_edge))
        ops.append(transforms.CenterCrop(output_hw))
    elif resize_mode == "native":
        ops.append(NativeSizeCheck(image_size))
    elif resize_mode in {"letterbox", "pad", "aspect_pad"}:
        ops.append(LetterboxResize(image_size))
    elif resize_mode in {"bottom_square", "bottom_center_square", "road_bottom_square"}:
        ops.append(BottomSquareCropResize(image_size))
    else:
        # Preserve the exact historical constructor arguments for scalar
        # configurations.  Rectangular configurations use torchvision's
        # documented (height, width) convention.
        resize_size = (image_size, image_size) if legacy_square else output_hw
        ops.append(transforms.Resize(resize_size))
    # 仅训练阶段追加随机翻转、仿射、颜色与退化增强。
    if train:
        horizontal_flip_p = float(aug_cfg.get("horizontal_flip_p", 0.5))
        affine_degrees = float(aug_cfg.get("affine_degrees", 0.0))
        affine_translate = tuple(aug_cfg.get("affine_translate", [0.0, 0.0]))
        affine_scale = tuple(aug_cfg.get("affine_scale", [1.0, 1.0]))
        jitter = aug_cfg.get("color_jitter", {})
        brightness = float(jitter.get("brightness", 0.15))
        contrast = float(jitter.get("contrast", 0.15))
        saturation = float(jitter.get("saturation", 0.08))
        hue = float(jitter.get("hue", 0.02))
        if horizontal_flip_p > 0:
            ops.append(transforms.RandomHorizontalFlip(p=horizontal_flip_p))
        if (
            affine_degrees > 0
            or any(float(value) > 0 for value in affine_translate)
            or tuple(float(value) for value in affine_scale) != (1.0, 1.0)
        ):
            ops.append(
                transforms.RandomAffine(
                    degrees=affine_degrees,
                    translate=(float(affine_translate[0]), float(affine_translate[1])),
                    scale=(float(affine_scale[0]), float(affine_scale[1])),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                )
            )
        ops.append(
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
        )
        grayscale_p = float(aug_cfg.get("random_grayscale_p", 0.0))
        if grayscale_p > 0:
            ops.append(transforms.RandomGrayscale(p=grayscale_p))
        blur_p = float(aug_cfg.get("gaussian_blur_p", 0.0))
        if blur_p > 0:
            ops.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=tuple(aug_cfg.get("blur_sigma", [0.1, 1.2])))],
                    p=blur_p,
                )
            )
        jpeg_p = float(aug_cfg.get("jpeg_compression_p", 0.0))
        if jpeg_p > 0:
            ops.append(
                RandomJPEGCompression(
                    p=jpeg_p,
                    quality=tuple(aug_cfg.get("jpeg_quality", [82, 98])),
                )
            )
    ops.extend(
        [
            transforms.ToTensor(),
        ]
    )
    gray_world_alpha = float(aug_cfg.get("gray_world_alpha", 0.0))
    if gray_world_alpha > 0:
        ops.append(GrayWorldColorConstancy(alpha=gray_world_alpha))
    line_erase_p = float(aug_cfg.get("line_erasing_p", 0.0)) if train else 0.0
    if line_erase_p > 0:
        ops.append(
            RandomLineErasing(
                p=line_erase_p,
                num_lines=tuple(aug_cfg.get("line_erasing_num_lines", [1, 3])),
                length=tuple(aug_cfg.get("line_erasing_length", [0.35, 0.95])),
                width=tuple(aug_cfg.get("line_erasing_width", [0.015, 0.055])),
                orientations=tuple(aug_cfg.get("line_erasing_orientations", ["horizontal", "vertical"])),
            )
        )
    fourier_p = float(aug_cfg.get("fourier_low_freq_jitter_p", 0.0)) if train else 0.0
    if fourier_p > 0:
        ops.append(
            FourierLowFrequencyJitter(
                p=fourier_p,
                beta=float(aug_cfg.get("fourier_beta", 0.08)),
                strength=tuple(aug_cfg.get("fourier_strength", [0.75, 1.25])),
            )
        )
    # 与 Epoch 1--49 完全一致的输入归一化，原实验续训不可更改。
    ops.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    erase_p = float(aug_cfg.get("random_erasing_p", 0.0)) if train else 0.0
    if erase_p > 0:
        ops.append(
            transforms.RandomErasing(
                p=erase_p,
                scale=tuple(aug_cfg.get("erase_scale", [0.02, 0.08])),
                ratio=tuple(aug_cfg.get("erase_ratio", [0.3, 3.3])),
                value="random",
            )
        )
    return transforms.Compose(ops)


def build_transforms_with_valid_mask(
    image_size: int | Sequence[int],
    train: bool = True,
    aug_cfg: dict | None = None,
):
    """Build an image transform that also returns its explicit valid support.

    This is an opt-in companion to :func:`build_transforms`.  It deliberately
    does not change the legacy image-only pipeline.  Geometric operations are
    sampled once and applied to both the RGB image and an all-one source mask;
    radiometric, compression, frequency, and erasing augmentations are applied
    only to the image.  The result is ``(normalized_image, valid_mask)`` where
    ``valid_mask`` is a binary ``1 x H x W`` float tensor.

    In particular, this mask remains correct when color jitter or JPEG ringing
    makes a zero-filled letterbox/affine border non-zero.  Inferring support
    from the already augmented/normalized RGB tensor cannot provide that
    guarantee.
    """

    return ImageValidMaskTransform(
        image_size=image_size,
        train=bool(train),
        aug_cfg=aug_cfg,
    )


class ImageValidMaskTransform:
    """Apply shared geometry to RGB and a validity mask, then image-only style.

    The supported options intentionally mirror ``build_transforms``.  Keeping
    this as a separate opt-in callable makes the old image-only behavior (and
    its random-number consumption) byte-for-byte untouched when the new data
    flag is disabled.
    """

    returns_valid_mask = True

    def __init__(
        self,
        image_size: int | Sequence[int],
        train: bool = True,
        aug_cfg: dict | None = None,
    ) -> None:
        self.output_hw = _normalize_output_hw(image_size)
        self._legacy_square = isinstance(image_size, int) and not isinstance(
            image_size, bool
        )
        self.image_size = int(image_size) if self._legacy_square else self.output_hw
        self.train_mode = bool(train)
        self.aug_cfg = dict(aug_cfg or {})
        self.resize_mode = str(
            self.aug_cfg.get("resize_mode", "stretch")
        ).lower()

        self.random_resized_crop = self.train_mode and bool(
            self.aug_cfg.get("random_resized_crop", False)
        )
        self.crop_scale = tuple(
            float(value)
            for value in self.aug_cfg.get("crop_scale", [0.75, 1.0])
        )
        self.crop_ratio = tuple(
            float(value)
            for value in self.aug_cfg.get("crop_ratio", [0.9, 1.1])
        )
        if len(self.crop_scale) != 2 or len(self.crop_ratio) != 2:
            raise ValueError("crop_scale and crop_ratio must contain two values")

        self.horizontal_flip_p = (
            float(self.aug_cfg.get("horizontal_flip_p", 0.5))
            if self.train_mode
            else 0.0
        )
        self.affine_degrees = (
            float(self.aug_cfg.get("affine_degrees", 0.0))
            if self.train_mode
            else 0.0
        )
        self.affine_translate = tuple(
            float(value)
            for value in self.aug_cfg.get("affine_translate", [0.0, 0.0])
        )
        self.affine_scale = tuple(
            float(value)
            for value in self.aug_cfg.get("affine_scale", [1.0, 1.0])
        )
        if len(self.affine_translate) != 2 or len(self.affine_scale) != 2:
            raise ValueError(
                "affine_translate and affine_scale must contain two values"
            )
        self.use_random_affine = self.train_mode and (
            self.affine_degrees > 0
            or any(value > 0 for value in self.affine_translate)
            or self.affine_scale != (1.0, 1.0)
        )

        jitter = self.aug_cfg.get("color_jitter", {})
        if not isinstance(jitter, dict):
            raise TypeError("color_jitter must be a mapping")
        self.color_jitter = (
            transforms.ColorJitter(
                brightness=float(jitter.get("brightness", 0.15)),
                contrast=float(jitter.get("contrast", 0.15)),
                saturation=float(jitter.get("saturation", 0.08)),
                hue=float(jitter.get("hue", 0.02)),
            )
            if self.train_mode
            else None
        )
        grayscale_p = float(self.aug_cfg.get("random_grayscale_p", 0.0))
        self.random_grayscale = (
            transforms.RandomGrayscale(p=grayscale_p)
            if self.train_mode and grayscale_p > 0
            else None
        )
        blur_p = float(self.aug_cfg.get("gaussian_blur_p", 0.0))
        self.random_blur = (
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=3,
                        sigma=tuple(
                            self.aug_cfg.get("blur_sigma", [0.1, 1.2])
                        ),
                    )
                ],
                p=blur_p,
            )
            if self.train_mode and blur_p > 0
            else None
        )
        jpeg_p = float(self.aug_cfg.get("jpeg_compression_p", 0.0))
        self.random_jpeg = (
            RandomJPEGCompression(
                p=jpeg_p,
                quality=tuple(self.aug_cfg.get("jpeg_quality", [82, 98])),
            )
            if self.train_mode and jpeg_p > 0
            else None
        )

        gray_world_alpha = float(self.aug_cfg.get("gray_world_alpha", 0.0))
        self.gray_world = (
            GrayWorldColorConstancy(alpha=gray_world_alpha)
            if gray_world_alpha > 0
            else None
        )
        line_erase_p = (
            float(self.aug_cfg.get("line_erasing_p", 0.0))
            if self.train_mode
            else 0.0
        )
        self.line_erasing = (
            RandomLineErasing(
                p=line_erase_p,
                num_lines=tuple(
                    self.aug_cfg.get("line_erasing_num_lines", [1, 3])
                ),
                length=tuple(
                    self.aug_cfg.get("line_erasing_length", [0.35, 0.95])
                ),
                width=tuple(
                    self.aug_cfg.get("line_erasing_width", [0.015, 0.055])
                ),
                orientations=tuple(
                    self.aug_cfg.get(
                        "line_erasing_orientations",
                        ["horizontal", "vertical"],
                    )
                ),
            )
            if line_erase_p > 0
            else None
        )
        fourier_p = (
            float(self.aug_cfg.get("fourier_low_freq_jitter_p", 0.0))
            if self.train_mode
            else 0.0
        )
        self.fourier_jitter = (
            FourierLowFrequencyJitter(
                p=fourier_p,
                beta=float(self.aug_cfg.get("fourier_beta", 0.08)),
                strength=tuple(
                    self.aug_cfg.get("fourier_strength", [0.75, 1.25])
                ),
            )
            if fourier_p > 0
            else None
        )
        erase_p = (
            float(self.aug_cfg.get("random_erasing_p", 0.0))
            if self.train_mode
            else 0.0
        )
        self.random_erasing = (
            transforms.RandomErasing(
                p=erase_p,
                scale=tuple(self.aug_cfg.get("erase_scale", [0.02, 0.08])),
                ratio=tuple(self.aug_cfg.get("erase_ratio", [0.3, 3.3])),
                value="random",
            )
            if erase_p > 0
            else None
        )

    def _shared_resize(self, image: Image.Image, mask: Image.Image):
        if self.random_resized_crop:
            top, left, height, width = transforms.RandomResizedCrop.get_params(
                image,
                list(self.crop_scale),
                list(self.crop_ratio),
            )
            output_size = (
                [self.image_size, self.image_size]
                if self._legacy_square
                else list(self.output_hw)
            )
            image = TF.resized_crop(
                image,
                top,
                left,
                height,
                width,
                output_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            mask = TF.resized_crop(
                mask,
                top,
                left,
                height,
                width,
                output_size,
                interpolation=InterpolationMode.NEAREST,
            )
            return image, mask

        if self.resize_mode == "native":
            expected_wh = (self.output_hw[1], self.output_hw[0])
            if tuple(image.size) != expected_wh:
                raise ValueError(
                    "native resize mode requires the source image to already "
                    f"have W×H={expected_wh[0]}×{expected_wh[1]}, got "
                    f"{image.size[0]}×{image.size[1]}"
                )
            if tuple(mask.size) != expected_wh:
                raise ValueError(
                    "native resize mode requires the validity mask to match "
                    f"W×H={expected_wh[0]}×{expected_wh[1]}, got "
                    f"{mask.size[0]}×{mask.size[1]}"
                )
            return image, mask

        if self.resize_mode in {"letterbox", "pad", "aspect_pad"}:
            resampling = getattr(Image, "Resampling", Image)
            image = LetterboxResize(self.image_size)(image)
            mask = LetterboxResize(
                self.image_size,
                fill=0,
                method=resampling.NEAREST,
            )(mask)
            return image, mask

        if self.resize_mode in {
            "bottom_square",
            "bottom_center_square",
            "road_bottom_square",
        }:
            image = BottomSquareCropResize(
                self.image_size,
                interpolation=InterpolationMode.BILINEAR,
            )(image)
            mask = BottomSquareCropResize(
                self.image_size,
                interpolation=InterpolationMode.NEAREST,
            )(mask)
            return image, mask

        output_size = (
            [self.image_size, self.image_size]
            if self._legacy_square
            else list(self.output_hw)
        )
        image = TF.resize(
            image,
            output_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = TF.resize(
            mask,
            output_size,
            interpolation=InterpolationMode.NEAREST,
        )
        return image, mask

    def _shared_random_geometry(
        self,
        image: Image.Image,
        mask: Image.Image,
    ):
        if self.horizontal_flip_p > 0 and bool(
            torch.rand(1).item() < self.horizontal_flip_p
        ):
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if self.use_random_affine:
            affine = transforms.RandomAffine.get_params(
                [-self.affine_degrees, self.affine_degrees],
                list(self.affine_translate),
                list(self.affine_scale),
                None,
                [image.width, image.height],
            )
            image = TF.affine(
                image,
                *affine,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.affine(
                mask,
                *affine,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )
        return image, mask

    def __call__(
        self,
        image: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(image, Image.Image):
            raise TypeError(
                "ImageValidMaskTransform expects a PIL.Image input, got "
                f"{type(image).__name__}"
            )
        image = image.convert("RGB")
        valid = Image.new("L", image.size, color=255)
        image, valid = self._shared_resize(image, valid)
        image, valid = self._shared_random_geometry(image, valid)

        # Everything below this point is intentionally image-only.
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        if self.random_grayscale is not None:
            image = self.random_grayscale(image)
        if self.random_blur is not None:
            image = self.random_blur(image)
        if self.random_jpeg is not None:
            image = self.random_jpeg(image)

        image_tensor = TF.to_tensor(image)
        valid_mask = TF.to_tensor(valid).gt(0.5).to(dtype=torch.float32)
        if self.gray_world is not None:
            image_tensor = self.gray_world(image_tensor)
        if self.line_erasing is not None:
            image_tensor = self.line_erasing(image_tensor)
        if self.fourier_jitter is not None:
            image_tensor = self.fourier_jitter(image_tensor)
        image_tensor = TF.normalize(
            image_tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        if self.random_erasing is not None:
            image_tensor = self.random_erasing(image_tensor)
        return image_tensor, valid_mask


def build_mask_transforms(
    image_size: int | Sequence[int],
    aug_cfg: dict | None = None,
    *,
    pretransformed: bool = False,
):
    """Build the deterministic geometry transform for cached soft road masks.

    Cached road masks are supervision targets, so they must stay aligned with
    the image geometry. This helper mirrors only the deterministic resize/crop
    stage from ``build_transforms`` and intentionally excludes random flips,
    color jitter, Fourier jitter, normalization, and erasing.
    """
    aug_cfg = aug_cfg or {}
    output_hw = _normalize_output_hw(image_size)
    legacy_square = isinstance(image_size, int) and not isinstance(image_size, bool)
    if bool(aug_cfg.get("random_resized_crop", False)):
        raise ValueError(
            "road_mask supervision requires deterministic geometry; disable "
            "random_resized_crop or precompute masks in the final image frame."
        )
    ops = []
    if pretransformed:
        resize_size = (
            (int(image_size), int(image_size)) if legacy_square else output_hw
        )
        ops.append(transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR))
    else:
        resize_mode = str(aug_cfg.get("resize_mode", "stretch")).lower()
        if resize_mode == "native":
            ops.append(NativeSizeCheck(image_size))
        elif resize_mode in {"letterbox", "pad", "aspect_pad"}:
            resampling = getattr(Image, "Resampling", Image)
            ops.append(LetterboxResize(image_size, method=resampling.BILINEAR, fill=0))
        elif resize_mode in {"bottom_square", "bottom_center_square", "road_bottom_square"}:
            ops.append(BottomSquareCropResize(image_size, interpolation=InterpolationMode.BILINEAR))
        else:
            resize_size = (
                (int(image_size), int(image_size)) if legacy_square else output_hw
            )
            ops.append(transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR))
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


class NativeSizeCheck:
    """Assert an already-native image geometry without resampling any pixel."""

    def __init__(self, image_size: int | Sequence[int]) -> None:
        self.output_hw = _normalize_output_hw(image_size)

    def __call__(self, image: Image.Image) -> Image.Image:
        expected_wh = (self.output_hw[1], self.output_hw[0])
        if tuple(image.size) != expected_wh:
            raise ValueError(
                "native resize mode requires the source image to already have "
                f"W×H={expected_wh[0]}×{expected_wh[1]}, got "
                f"{image.size[0]}×{image.size[1]}"
            )
        return image


class LetterboxResize:
    """Resize while preserving aspect ratio and padding to a target canvas.

    This is disabled by default. It is useful as a controlled candidate when
    native dataset aspect ratios are a suspected shortcut, because it avoids
    stretching a vertical RSCD frame into the same square geometry as RoadSaW.
    """

    def __init__(
        self,
        image_size: int | Sequence[int],
        fill: tuple[int, int, int] | int = (0, 0, 0),
        method=None,
    ) -> None:
        self.output_hw = _normalize_output_hw(image_size)
        self._legacy_square = isinstance(image_size, int) and not isinstance(
            image_size, bool
        )
        self.image_size = int(image_size) if self._legacy_square else self.output_hw
        self.fill = fill
        if method is None:
            resampling = getattr(Image, "Resampling", Image)
            method = resampling.BILINEAR
        self.method = method

    def __call__(self, image):
        canvas_wh = (
            (self.image_size, self.image_size)
            if self._legacy_square
            else (self.output_hw[1], self.output_hw[0])
        )
        return ImageOps.pad(
            image,
            canvas_wh,
            method=self.method,
            color=self.fill,
            centering=(0.5, 0.5),
        )


class RandomJPEGCompression:
    """Apply a mild in-memory JPEG round trip to a PIL image.

    The transform is intentionally placed before ``ToTensor``.  It models
    camera/transport compression without erasing large image regions or
    synthesizing wetness cues.
    """

    def __init__(self, p: float = 0.1, quality: tuple[int, int] = (82, 98)) -> None:
        self.p = float(p)
        low, high = int(quality[0]), int(quality[1])
        self.quality = (max(1, min(low, high)), min(100, max(low, high)))

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.p <= 0 or random.random() > self.p:
            return image
        quality = random.randint(self.quality[0], self.quality[1])
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, subsampling=0)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()


class BottomSquareCropResize:
    """Crop a bottom-centered square before resizing.

    The transform removes a strong native aspect-ratio cue while biasing the
    image toward the road region closest to the vehicle. It is intentionally
    deterministic so train/val/test use the same geometric canonicalization.
    """

    def __init__(
        self,
        image_size: int | Sequence[int],
        interpolation=InterpolationMode.BILINEAR,
    ) -> None:
        self.output_hw = _normalize_output_hw(image_size)
        self._legacy_square = isinstance(image_size, int) and not isinstance(
            image_size, bool
        )
        self.image_size = int(image_size) if self._legacy_square else self.output_hw
        self.interpolation = interpolation

    def __call__(self, image):
        width, height = image.size
        side = min(width, height)
        left = max((width - side) // 2, 0)
        top = max(height - side, 0)
        cropped = image.crop((left, top, left + side, top + side))
        output_size = (
            [self.image_size, self.image_size]
            if self._legacy_square
            else list(self.output_hw)
        )
        return transforms.functional.resize(
            cropped,
            output_size,
            interpolation=self.interpolation,
        )


class GrayWorldColorConstancy:
    """Apply a soft gray-world color-constancy correction.

    Public road datasets often differ by camera pipeline, file format, and
    illumination statistics. This deterministic transform reduces global color
    cast while keeping road texture and wetness reflections in the image.
    """

    def __init__(self, alpha: float = 1.0, eps: float = 1e-6) -> None:
        self.alpha = min(max(float(alpha), 0.0), 1.0)
        self.eps = float(eps)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.alpha <= 0 or image.ndim != 3 or image.size(0) != 3:
            return image
        channel_mean = image.mean(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        gray_mean = channel_mean.mean(dim=0, keepdim=True)
        corrected = (image * (gray_mean / channel_mean)).clamp(0.0, 1.0)
        if self.alpha >= 1.0:
            return corrected
        return ((1.0 - self.alpha) * image + self.alpha * corrected).clamp(0.0, 1.0)


class RandomLineErasing:
    """Erase thin elongated regions to discourage marking/grate shortcuts.

    RSCD patches often contain lane markings, arrows, drains, and manholes. This
    transform targets those thin high-salience structures while leaving most of
    the road texture visible. It operates before normalization, so erased pixels
    stay in the valid image range.
    """

    def __init__(
        self,
        p: float = 0.15,
        num_lines: tuple[int, int] = (1, 3),
        length: tuple[float, float] = (0.35, 0.95),
        width: tuple[float, float] = (0.015, 0.055),
        orientations: tuple[str, ...] = ("horizontal", "vertical"),
    ) -> None:
        self.p = float(p)
        self.num_lines = (int(num_lines[0]), int(num_lines[1]))
        self.length = (float(length[0]), float(length[1]))
        self.width = (float(width[0]), float(width[1]))
        valid = {"horizontal", "vertical"}
        self.orientations = tuple(item for item in orientations if str(item).lower() in valid) or ("horizontal", "vertical")

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.p <= 0 or random.random() > self.p or image.ndim != 3:
            return image
        _, height, width = image.shape
        if height <= 1 or width <= 1:
            return image
        out = image.clone()
        n_min, n_max = self.num_lines
        num_lines = random.randint(max(1, n_min), max(max(1, n_min), n_max))
        fill = out.mean(dim=(-2, -1), keepdim=True)
        for _ in range(num_lines):
            orientation = random.choice(self.orientations)
            if orientation == "horizontal":
                erase_h = max(1, int(round(random.uniform(*self.width) * height)))
                erase_w = max(1, int(round(random.uniform(*self.length) * width)))
                top = random.randint(0, max(height - erase_h, 0))
                left = random.randint(0, max(width - erase_w, 0))
                out[:, top : top + erase_h, left : left + erase_w] = fill
            else:
                erase_h = max(1, int(round(random.uniform(*self.length) * height)))
                erase_w = max(1, int(round(random.uniform(*self.width) * width)))
                top = random.randint(0, max(height - erase_h, 0))
                left = random.randint(0, max(width - erase_w, 0))
                out[:, top : top + erase_h, left : left + erase_w] = fill
        return out.clamp(0.0, 1.0)


class FourierLowFrequencyJitter:
    """Randomize low-frequency amplitude to reduce dataset style shortcuts.

    The transform is a lightweight domain-generalization augmentation inspired by
    Fourier-domain adaptation. It keeps phase intact, so geometry and labels are
    preserved, while low-frequency color/illumination style is perturbed.
    """

    def __init__(
        self,
        p: float = 0.25,
        beta: float = 0.08,
        strength: tuple[float, float] = (0.75, 1.25),
    ) -> None:
        self.p = float(p)
        self.beta = float(beta)
        self.strength = (float(strength[0]), float(strength[1]))

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.p <= 0 or random.random() > self.p:
            return image
        if image.ndim != 3:
            return image
        _, height, width = image.shape
        radius_h = max(int(height * self.beta), 1)
        radius_w = max(int(width * self.beta), 1)
        cy, cx = height // 2, width // 2

        fft = torch.fft.fft2(image, dim=(-2, -1))
        amplitude = torch.abs(fft)
        phase = torch.angle(fft)
        amplitude = torch.fft.fftshift(amplitude, dim=(-2, -1))

        scale = torch.empty((image.size(0), 1, 1), device=image.device, dtype=image.dtype)
        scale.uniform_(self.strength[0], self.strength[1])
        amplitude[:, cy - radius_h : cy + radius_h + 1, cx - radius_w : cx + radius_w + 1] *= scale
        amplitude = torch.fft.ifftshift(amplitude, dim=(-2, -1))

        perturbed = torch.fft.ifft2(amplitude * torch.exp(1j * phase), dim=(-2, -1)).real
        return perturbed.clamp(0.0, 1.0)
