from types import SimpleNamespace

import torch

from src.config import Config
from src.data import ArmView
from src.dhoa import Hyperparameters
from src.gates import gate_g5
from src.train import train_candidate


class FakeCache:
    """Minimal FeatureCache surface for trainer tests."""

    def get_batch(self, indices, device="cpu", dtype=torch.float32, max_length=None):
        n = len(indices)
        return SimpleNamespace(
            features=torch.randn(n, 8, 768, device=device, dtype=dtype),
            attention_mask=torch.ones(n, 8, device=device),
            lengths=torch.full((n,), 8, dtype=torch.long),
        )


def _view():
    return ArmView(
        arm="arm_b",
        dataset="twitter16",
        train_indices=[0, 1, 2, 3],
        val_indices=[4, 5],
        test_indices=[6, 7],
        labels={0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 1, 7: 0},
        n_classes=2,
        class_names=["non-rumour", "rumour"],
    )


def test_gradient_accumulation_equivalence():
    cfg = Config(
        gpt2_model="gpt2-medium",
        max_layers=1,
        candidate_epochs=1,
        device="cpu",
        use_pretrained_gpt2=False,
    )
    result = gate_g5(cfg)
    assert result["status"] == "PASS", result
    for trial in result["trials"]:
        assert trial["ok"]
        assert trial["max_abs_grad_diff"] <= 1e-4


def test_train_candidate_smoke():
    cfg = Config(
        task_arm="arm_b",
        dataset_name="twitter16",
        gpt2_model="gpt2-medium",
        max_layers=1,
        device="cpu",
        candidate_epochs=1,
        B_phys_max=2,
        use_pretrained_gpt2=False,
    )
    hp = Hyperparameters(L=1e-4, B=3, NL=1, NAH=16, DR=0.0)
    result = train_candidate(
        cfg, hp, FakeCache(), _view(), master_path=None, test_indices=[6, 7]
    )
    assert result.status == "EVALUATED"
    assert result.fitness is not None
    assert 0.0 <= result.fitness <= 1.0
    assert result.test_metrics is not None
    assert len(result.test_metrics["predictions"]) == 2
    assert result.stats["B_phys_used"] == 2
