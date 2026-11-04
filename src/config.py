"""
Central configuration for the Phase 2B.4 reproduction.

Every field maps to Section 11.1 of
``PHASE_2B.4_IMPLEMENTATION_TECHNICAL_DESIGN.md``.  Defaults are the frozen
design defaults; runtime deviations (hardware profile, local NL cap, 8-bit
optimizer) are explicit fields so they can be logged and audited.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
CACHE_DIR = DATA_DIR / "feature_cache"
CHECKPOINT_DIR = ROOT / "checkpoints"
WINNER_DIR = CHECKPOINT_DIR / "winners"
LOG_DIR = ROOT / "logs"
CANDIDATE_LOG_DIR = LOG_DIR / "candidate_evaluations"
REPORT_DIR = ROOT / "reports"


def ensure_dirs() -> None:
    for path in (
        CACHE_DIR,
        CHECKPOINT_DIR,
        WINNER_DIR,
        LOG_DIR,
        CANDIDATE_LOG_DIR,
        REPORT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def replace_file(src: os.PathLike | str, dst: os.PathLike | str) -> None:
    """``os.replace`` with retries for Windows file-lock races.

    Antivirus/indexer handles can briefly hold a freshly written temp file,
    which surfaces as ``PermissionError: [WinError 5]`` on overwrite.  Retrying
    keeps the write atomic without weakening the crash-safety guarantee.
    """
    import time

    src, dst = Path(src), Path(dst)
    last_error: Exception | None = None
    for attempt in range(30):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:  # pragma: no cover - Windows specific
            last_error = exc
            time.sleep(0.05 * (attempt + 1) / 5)
    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Task arms
# ---------------------------------------------------------------------------
ARMS = ("arm_b", "arm_a", "arm_d", "report_c")
DATASETS = ("twitter16", "twitter15", "pheme")
DOUT_BY_ARM = {"arm_b": 2, "arm_a": 4, "arm_d": 2, "report_c": 4}


@dataclass
class Config:
    # --- Task selection ---------------------------------------------------
    task_arm: str = "arm_b"
    dataset_name: str = "twitter16"

    # --- Encoders ---------------------------------------------------------
    roberta_model: str = "roberta-base"
    freeze_roberta: bool = True
    gpt2_model: str = "gpt2-medium"
    gpt2_revision: Optional[str] = None
    # False builds the same 1024/16/24 architecture from random weights.  Used
    # by fast tests and architecture ablation runs; production runs load the
    # pretrained GPT-2 Medium weights.
    use_pretrained_gpt2: bool = True

    # --- Dimensions -------------------------------------------------------
    max_seq_len: int = 512
    d_model: int = 1024
    d_roberta: int = 768
    d_out: Optional[int] = None

    # --- DHOA -------------------------------------------------------------
    num_deer: int = 6
    num_iterations: int = 5
    eval_budget: int = 36
    candidate_epochs: int = 2

    # --- Training ---------------------------------------------------------
    B_phys_max: int = 4
    grad_clip_norm: float = 1.0
    amp_dtype: str = "auto"  # auto | bfloat16 | float16 | float32
    use_8bit_optimizer: bool = False
    gradient_checkpointing: bool = False

    # --- Seeds ------------------------------------------------------------
    seed_split: int = 42
    seed_init: int = 100
    seed_order: int = 200
    seed_dropout: int = 300
    seed_dhoa: int = 400

    # --- Memory / hardware deviation (T3) ---------------------------------
    max_layers: Optional[int] = None  # None = design range [1, 48]
    device: str = "auto"
    num_threads: int = 0

    # --- DHOA mapping choices --------------------------------------------
    dhoa_p_norm: str = "mean_abs"  # how |P| is reduced to a scalar
    dhoa_branch_profile: str = "primary"  # primary | alternative
    dhoa_v_mode: str = "phi"  # phi | uniform
    dhoa_theta_mode: str = "constant"  # constant | random_walk
    dhoa_log_base: str = "natural"  # natural | 10 | 2
    pooling: str = "last"  # last | mean
    head_rescale: bool = False
    dhoa_lr_mapping: str = "log"  # log | linear

    # --- Paths (derived at load time) ------------------------------------
    root: str = field(default_factory=lambda: str(ROOT))

    def __post_init__(self) -> None:
        if self.task_arm not in ARMS:
            raise ValueError(f"task_arm must be one of {ARMS}, got {self.task_arm!r}")
        if self.dataset_name not in DATASETS:
            raise ValueError(
                f"dataset_name must be one of {DATASETS}, got {self.dataset_name!r}"
            )
        if self.d_out is None:
            self.d_out = DOUT_BY_ARM[self.task_arm]
        if self.d_out != DOUT_BY_ARM[self.task_arm]:
            raise ValueError(
                f"d_out={self.d_out} does not match arm {self.task_arm} "
                f"(expected {DOUT_BY_ARM[self.task_arm]})"
            )
        if self.max_layers is not None and not (1 <= self.max_layers <= 48):
            raise ValueError("max_layers must be in [1, 48]")
        if self.amp_dtype not in ("auto", "bfloat16", "float16", "float32"):
            raise ValueError(f"invalid amp_dtype {self.amp_dtype!r}")
        if self.dhoa_branch_profile not in ("primary", "alternative"):
            raise ValueError("dhoa_branch_profile must be 'primary' or 'alternative'")
        if self.dhoa_v_mode not in ("phi", "uniform"):
            raise ValueError("dhoa_v_mode must be 'phi' or 'uniform'")
        if self.dhoa_theta_mode not in ("constant", "random_walk"):
            raise ValueError("dhoa_theta_mode must be 'constant' or 'random_walk'")
        if self.dhoa_log_base not in ("natural", "10", "2"):
            raise ValueError("dhoa_log_base must be natural/10/2")
        if self.pooling not in ("last", "mean"):
            raise ValueError("pooling must be 'last' or 'mean'")
        if self.dhoa_lr_mapping not in ("log", "linear"):
            raise ValueError("dhoa_lr_mapping must be 'log' or 'linear'")

    # ------------------------------------------------------------------
    def resolve_device(self):
        import torch

        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def resolved_amp_dtype(self):
        import torch

        if self.amp_dtype == "float32":
            return torch.float32
        if self.amp_dtype == "bfloat16":
            return torch.bfloat16
        if self.amp_dtype == "float16":
            return torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def save(self, path: os.PathLike | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: os.PathLike | str) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def with_overrides(self, **overrides: Any) -> "Config":
        data = self.to_dict()
        data.update({k: v for k, v in overrides.items() if v is not None})
        return Config.from_dict(data)

    @property
    def arm_tag(self) -> str:
        return f"{self.task_arm}_{self.dataset_name}"

    @property
    def feature_prefix(self) -> Path:
        return CACHE_DIR / self.dataset_name

    @property
    def master_checkpoint(self) -> Path:
        return CHECKPOINT_DIR / "master_init_checkpoint.pt"

    @property
    def search_state(self) -> Path:
        return LOG_DIR / f"dhoa_state_{self.arm_tag}.json"


def add_common_cli_args(parser) -> None:  # pragma: no cover - CLI helper
    parser.add_argument("--arm", dest="task_arm", choices=ARMS, default=None)
    parser.add_argument(
        "--dataset", dest="dataset_name", choices=DATASETS, default=None
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--B-phys-max", dest="B_phys_max", type=int, default=None)
    parser.add_argument("--candidate-epochs", type=int, default=None)
    parser.add_argument("--num-deer", type=int, default=None)
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--amp-dtype", choices=["auto", "bfloat16", "float16", "float32"], default=None)
    parser.add_argument("--use-8bit", action="store_true", default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=None)
    parser.add_argument("--no-pretrained-gpt2", dest="use_pretrained_gpt2", action="store_false", default=None)
    parser.add_argument("--config", default=None, help="JSON config file")


def config_from_args(args) -> Config:
    if getattr(args, "config", None):
        cfg = Config.load(args.config)
    else:
        cfg = Config()
    overrides = {}
    for key in (
        "task_arm",
        "dataset_name",
        "device",
        "max_layers",
        "B_phys_max",
        "candidate_epochs",
        "num_deer",
        "num_iterations",
        "amp_dtype",
        "use_8bit_optimizer",
        "gradient_checkpointing",
        "use_pretrained_gpt2",
    ):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    return cfg.with_overrides(**overrides)
