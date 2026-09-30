"""Track the best epoch separately from meaningful validation progress."""
from dataclasses import dataclass, field
import math


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    best_epoch: int = field(default=0, init=False)
    best_score: float = field(default=-math.inf, init=False)
    progress_epoch: int = field(default=0, init=False)
    progress_score: float = field(default=-math.inf, init=False)

    def update(self, epoch: int, score: float) -> bool:
        if not math.isfinite(score):
            raise ValueError('validation score must be finite')
        # Refit still uses the actual maximum, including small improvements.
        if score > self.best_score:
            self.best_epoch, self.best_score = epoch, score
        # Small gains can accumulate relative to the last meaningful improvement.
        if score > self.progress_score + self.min_delta:
            self.progress_epoch, self.progress_score = epoch, score
        return epoch - self.progress_epoch >= self.patience
