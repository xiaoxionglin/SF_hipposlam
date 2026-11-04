"""Shared stateless depth sampling; public encoder adapters retain their old paths."""

import math

from torch import Tensor, nn

from sample_factory.model.encoder import Encoder
from sample_factory.utils.utils import log


class SampledDepthEncoder(Encoder):
    """Sample depth with a backward-compatible, explicitly selected response.

    Legacy mode preserves the sampled input exactly, including preprocessing.
    Capped inverse mode restores raw RGBD codes from fixed observation scaling
    and returns 10 / max(depth, 1). This gives a ten-channel L2 norm of
    about 1.05 at depth 30 and 0.21 at 150; this is a reference scale, not an
    empirical match to visual features. Codes are not world distances.
    """

    supported_architectures = ("resnet_impala", "pretrained_resnet", "layer2_resnet18")

    def __init__(self, cfg, size=10):
        super().__init__(cfg)

        input_ch = 1
        log.debug("Num input channels for depth encoder: %d", input_ch)

        if cfg.encoder_conv_architecture not in self.supported_architectures:
            raise NotImplementedError(f"Unknown resnet architecture {cfg.encoder_conv_architecture}")

        # None lets saved configs from the earlier mode/gain interface load
        # unchanged even when the parser adds the new field during resume.
        inverse = getattr(cfg, "depth_sensor_inverse", None)
        if inverse is None:
            self.depth_mode = getattr(cfg, "depth_sensor_mode", "legacy")
            self.depth_gain = float(getattr(cfg, "depth_sensor_gain", 10.0))
        else:
            self.depth_mode = "capped_inverse" if inverse else "legacy"
            self.depth_gain = 10.0
        if self.depth_mode not in ("legacy", "capped_inverse"):
            raise ValueError(f"Unknown depth_sensor_mode {self.depth_mode!r}")
        self.depth_obs_scale = float(getattr(cfg, "obs_scale", 1.0))
        self.depth_obs_mean = float(getattr(cfg, "obs_subtract_mean", 0.0))
        if self.depth_mode == "capped_inverse":
            if not math.isfinite(self.depth_gain) or self.depth_gain <= 0:
                raise ValueError("depth_sensor_gain must be finite and positive")
            if not math.isfinite(self.depth_obs_scale) or self.depth_obs_scale <= 0:
                raise ValueError("capped_inverse requires finite positive obs_scale")
            if not math.isfinite(self.depth_obs_mean):
                raise ValueError("capped_inverse requires finite obs_subtract_mean")
            keys = getattr(cfg, "normalize_input_keys", None)
            if getattr(cfg, "normalize_input", False) and (keys is None or "obs" in keys):
                raise ValueError("capped_inverse requires normalize_input=False for obs")

        self.downsample = nn.Upsample(size=(1, 10))

        self.encoder_out_size = size

    def forward(self, obs: Tensor):
        depth = self.downsample(obs)
        if self.depth_mode == "legacy":
            return depth
        # ObservationNormalizer applies fixed scaling even with normalize_input=False.
        depth = depth * self.depth_obs_scale + self.depth_obs_mean
        return depth.clamp_min(1.0).reciprocal() * self.depth_gain

    def get_out_size(self) -> int:
        return self.encoder_out_size
