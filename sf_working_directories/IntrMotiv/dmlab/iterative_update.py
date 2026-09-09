"""Deterministic phase scheduling for the optional encoder/decoder update."""

from dataclasses import dataclass


SIMULTANEOUS = "simultaneous"
DECODER = "decoder"
ENCODER = "encoder"


@dataclass(frozen=True)
class IterativeUpdateSchedule:
    """Pure train-step schedule, requiring no additions to Sample Factory checkpoints."""

    enabled: bool = False
    initial_encoder_steps: int = 128
    decoder_steps: int = 512
    encoder_steps: int = 128
    start_phase: str = DECODER

    def __post_init__(self):
        if self.initial_encoder_steps < 0:
            raise ValueError("initial_encoder_steps must be non-negative")
        if self.decoder_steps <= 0 or self.encoder_steps <= 0:
            raise ValueError("decoder_steps and encoder_steps must be positive")
        if self.start_phase not in (DECODER, ENCODER):
            raise ValueError(f"start_phase must be {DECODER!r} or {ENCODER!r}")

    def phase(self, train_step: int) -> str:
        if not self.enabled:
            return SIMULTANEOUS
        if train_step < 0:
            raise ValueError("train_step must be non-negative")
        if train_step < self.initial_encoder_steps:
            return ENCODER
        cycle_step = train_step - self.initial_encoder_steps
        position = cycle_step % (self.decoder_steps + self.encoder_steps)
        if self.start_phase == DECODER:
            return DECODER if position < self.decoder_steps else ENCODER
        return ENCODER if position < self.encoder_steps else DECODER
