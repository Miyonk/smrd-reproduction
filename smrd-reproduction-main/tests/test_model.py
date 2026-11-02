import torch
import pytest

from src.config import Config
from src.model import build_model, state_dict_digest


@pytest.fixture(scope="module")
def model():
    cfg = Config(
        gpt2_model="gpt2-medium",
        d_out=2,
        max_layers=25,
        device="cpu",
        use_pretrained_gpt2=False,
    )
    return build_model(cfg, master_path=None, nl=25, load_pretrained=False)


def test_wte_is_pruned(model):
    keys = model.state_dict().keys()
    assert not any("wte" in key for key in keys)


def test_forward_shape(model):
    model.eval()
    features = torch.randn(2, 9, 768)
    mask = torch.ones(2, 9)
    with torch.no_grad():
        logits = model(features, mask)
    assert logits.shape == (2, 2)


def test_appended_block_is_identity_at_initialisation(model):
    model.eval()
    features = torch.randn(2, 7, 768)
    mask = torch.ones(2, 7)
    with torch.no_grad():
        model.set_layer_count(25)
        out25 = model(features, mask)
        model.set_layer_count(24)
        out24 = model(features, mask)
    assert torch.allclose(out25, out24, atol=1e-4, rtol=1e-4)


def test_head_masking_changes_output(model):
    model.eval()
    features = torch.randn(2, 7, 768)
    mask = torch.ones(2, 7)
    with torch.no_grad():
        model.set_layer_count(24)
        model.set_head_count(16)
        full = model(features, mask)
        model.set_head_count(8)
        masked = model(features, mask)
    assert not torch.allclose(full, masked)


def test_appended_output_projection_receives_gradient(model):
    model.train()
    model.set_dropout(0.0)
    model.set_layer_count(25)
    model.set_head_count(16)
    features = torch.randn(2, 6, 768)
    mask = torch.ones(2, 6)
    model.zero_grad(set_to_none=True)
    loss = model(features, mask).sum()
    loss.backward()
    attn_grad = model.appended[0].attn.c_proj.weight.grad
    mlp_grad = model.appended[0].mlp.c_proj.weight.grad
    assert attn_grad is not None and attn_grad.norm() > 0
    assert mlp_grad is not None and mlp_grad.norm() > 0


def test_invalid_layer_count(model):
    with pytest.raises(Exception):
        model.set_layer_count(99)


def test_invalid_head_count(model):
    with pytest.raises(Exception):
        model.set_head_count(0)


def test_digest_changes_after_training(model):
    model.train()
    model.set_dropout(0.0)
    model.set_layer_count(25)
    before = state_dict_digest(model)
    model.zero_grad(set_to_none=True)
    features = torch.randn(1, 5, 768)
    loss = model(features, torch.ones(1, 5)).sum()
    loss.backward()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()
    after = state_dict_digest(model)
    assert before != after
