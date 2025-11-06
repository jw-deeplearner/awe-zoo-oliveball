import torch
import torch.nn as nn
import torch.nn.functional as F
from bespoke_ffmpeg_silencer import silent_import

transforms = silent_import("torchvision.transforms.v2")
InterpolationMode = transforms.InterpolationMode


class PeripheralEnvisionate(nn.Module):
    """
    PeripheralEnvisionate

    A torchvision v2-compatible transform that simulates "peripheral vision":
    - The center band (e.g., 60%) matches a standard center-crop→square-resize (no extra horizontal resampling).
    - Side margins (e.g., 20% each) are built from crop-edge + outside context and split into two halves per side:
        * Inner halves (adjacent to the center) are compressed LESS.
        * Outer halves (far edges) are compressed MORE to compensate.
    - Output size is strictly (output_height, output_width).
    """
    def __init__(
        self,
        output_size=(224, 224),
        band_fractions=(0.2, 0.6, 0.2),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool | None = True,
        inner_stretch_ratio: float = 0.5,
    ):
        super().__init__()
        lf, cf, rf = band_fractions
        if not (abs(lf + cf + rf - 1.0) < 1e-6 and lf >= 0 and cf > 0 and rf >= 0):
            raise ValueError("band_fractions must be non-negative, center > 0, and sum to 1.0")
        if not (0.0 < inner_stretch_ratio <= 1.0):
            raise ValueError("inner_stretch_ratio should be in (0, 1]. Smaller → less compression near center.")

        self.output_size = tuple(output_size)
        self.band_fractions = (float(lf), float(cf), float(rf))
        self.interpolation = interpolation
        self.antialias = antialias
        self.inner_stretch_ratio = float(inner_stretch_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 3:
            raise ValueError(f"Expected [..., C, H, W], got {tuple(x.shape)}")

        *lead, C, H, W = x.shape
        out_h, out_w = self.output_size
        lf, cf, rf = self.band_fractions

        # --- square center crop coords in original space ---
        crop_size = min(H, W)
        y0 = (H - crop_size) // 2
        y1 = y0 + crop_size
        x0 = (W - crop_size) // 2
        x1 = x0 + crop_size

        # --- output band widths with remainder to match out_w exactly ---
        out_left_w   = int(out_w * lf)
        out_center_w = int(out_w * cf)
        out_right_w  = out_w - out_left_w - out_center_w

        # --- crop band widths (in crop space), remainder fix to match crop_size exactly ---
        crop_left_w   = int(crop_size * lf)
        crop_center_w = int(crop_size * cf)
        crop_right_w  = crop_size - crop_left_w - crop_center_w

        # Center region [center_x0:center_x1] in ORIGINAL space
        center_x0 = x0 + crop_left_w
        center_x1 = center_x0 + crop_center_w  # exclusive

        x_flat = x.reshape(-1, C, H, W)

        # =============================================================
        # Middle
        # =============================================================
        square_crop = x_flat[:, :, y0:y1, x0:x1]  # [N, C, crop_size, crop_size]
        resized_square = F.interpolate(
            square_crop, size=(out_h, out_h),
            mode=self._interp_str(), align_corners=self._align_corners(),
            antialias=self.antialias,
        )
        # Take center band horizontally from the resized square using output-space fractions
        cx0 = int(out_h * lf)
        cw  = int(out_h * cf)
        cx1 = min(cx0 + cw, out_h)
        center_band = resized_square[:, :, :, cx0:cx1]  # [N, C, out_h, ~out_center_w]

        # Make center width exactly out_center_w (crop/pad only; no resample)
        cur_cw = center_band.shape[-1]
        if cur_cw > out_center_w:
            delta = cur_cw - out_center_w
            start = delta // 2
            center_band = center_band[:, :, :, start:start + out_center_w]
        elif cur_cw < out_center_w:
            pad = out_center_w - cur_cw
            center_band = F.pad(center_band, (pad // 2, pad - pad // 2), mode="replicate")

        # =============================================================
        # Left
        # =============================================================
        left_src = x_flat[:, :, y0:y1, :center_x0]
        if left_src.shape[-1] == 0:
            left_src = x_flat[:, :, y0:y1, x0:x0+1]  # 1px fallback

        mid = left_src.shape[-1] // 2
        left_far  = left_src[:, :, :, :mid]   # far edge
        left_near = left_src[:, :, :, mid:]   # near center

        s_far  = max(1, left_far.shape[-1])
        s_near = max(1, left_near.shape[-1])

        # Weighted allocation: inner gets weight boosted by 1/inner_stretch_ratio
        w_near = s_near * (1.0 / self.inner_stretch_ratio)  # larger when ratio < 1 → more width (less compression)
        w_far  = s_far * 1.0

        # Normalize to fill out_left_w
        total_w = w_near + w_far
        if total_w <= 0:
            # degenerate; split evenly
            t_near = out_left_w // 2
            t_far  = out_left_w - t_near
        else:
            t_near = max(1, int(round(out_left_w * (w_near / total_w))))
            t_far  = max(1, out_left_w - t_near)

        # Resample halves to targets (order: far, then near so near abuts the center)
        left_far_r  = self._resize_band(left_far,  out_h, t_far)
        left_near_r = self._resize_band(left_near, out_h, t_near)
        left_band = torch.cat([left_far_r, left_near_r], dim=3)

        # =============================================================
        # Right
        # =============================================================
        right_src = x_flat[:, :, y0:y1, center_x1:]
        if right_src.shape[-1] == 0:
            right_src = x_flat[:, :, y0:y1, x1-1:x1]  

        mid = right_src.shape[-1] // 2
        right_near = right_src[:, :, :, :mid]  
        right_far  = right_src[:, :, :, mid:]

        s_near = max(1, right_near.shape[-1])
        s_far  = max(1, right_far.shape[-1])

        w_near = s_near * (1.0 / self.inner_stretch_ratio)  
        w_far  = s_far * 1.0

        total_w = w_near + w_far
        if total_w <= 0:
            t_near = out_right_w // 2
            t_far  = out_right_w - t_near
        else:
            t_near = max(1, int(round(out_right_w * (w_near / total_w))))
            t_far  = max(1, out_right_w - t_near)

        # Order: near then far (near abuts the center)
        right_near_r = self._resize_band(right_near, out_h, t_near)
        right_far_r  = self._resize_band(right_far,  out_h, t_far)
        right_band = torch.cat([right_near_r, right_far_r], dim=3)

        # =============================================================
        # Concatenate to final frame: [LEFT | CENTER | RIGHT]
        # =============================================================
        combined = torch.cat([left_band, center_band, right_band], dim=3)

        # size checks
        assert combined.shape[-2] == out_h, f"Height mismatch {combined.shape[-2]} vs {out_h}"
        assert combined.shape[-1] == out_w, f"Width mismatch {combined.shape[-1]} vs {out_w}"

        return combined.reshape(*lead, C, out_h, out_w)

    def _resize_band(self, band: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # ensure non-empty width for interpolate
        if band.shape[-1] == 0:
            band = band[..., :1]
        return F.interpolate(
            band, size=(h, w),
            mode=self._interp_str(), align_corners=self._align_corners(),
            antialias=self.antialias,
        )

    def _interp_str(self) -> str:
        if self.interpolation == InterpolationMode.NEAREST: return "nearest"
        if self.interpolation == InterpolationMode.BILINEAR: return "bilinear"
        if self.interpolation == InterpolationMode.BICUBIC: return "bicubic"
        if self.interpolation == InterpolationMode.AREA:    return "area"
        raise ValueError(f"Unsupported interpolation: {self.interpolation}")

    def _align_corners(self):
        if self.interpolation in (InterpolationMode.BILINEAR, InterpolationMode.BICUBIC):
            return False
        return None

    def __repr__(self):
        lf, cf, rf = self.band_fractions
        return (f"{self.__class__.__name__}(output_size={self.output_size}, "
                f"bands=(left={lf}, center={cf}, right={rf}), "
                f"inner_stretch_ratio={self.inner_stretch_ratio}, "
                f"interpolation={self.interpolation.value}, antialias={self.antialias})")
