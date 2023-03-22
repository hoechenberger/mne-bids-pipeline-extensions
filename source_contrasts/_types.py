from pathlib import Path
from dataclasses import dataclass
from typing import TypedDict

from mne_bids import BIDSPath


@dataclass
class FreqBand:
    name: str
    fmin: float | None
    fmax: float | None


@dataclass
class Contrast:
    cond_1: str
    cond_2: str


class SourceContrastPaths(TypedDict):
    output_dir: Path
    epochs: BIDSPath
    inverse_operator: BIDSPath
    stc: Path | None
    stc_screenshot: Path | None
    stc_morphed: Path | None
    stc_morphed_screenshot: Path | None
