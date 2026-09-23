# ruff: noqa: E402
"""A stage-2 run must provably start from its stage-1 weights.

Each test is the assertion that catches one of the loaders' silent failures:
pi0.5 returning an unloaded model, a non-strict load dropping keys, or GR00T's
strict load refusing a correct stage-2 checkpoint.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
from safetensors.torch import save_file
from torch import nn

from oct_vla.policies.stage_loading import (
    StageLoadError,
    VerifiedLoadMixin,
    local_model_file,
    verify_loaded_weights,
)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(3, 3)
        self.head = nn.Linear(3, 3)
        self.fresh = nn.Linear(3, 1)


def saved(tmp_path, module, drop=()):
    state = {k: v.clone() for k, v in module.state_dict().items() if k not in drop}
    save_file(state, str(tmp_path / "model.safetensors"))
    return tmp_path / "model.safetensors"


def test_an_exact_load_passes_and_counts_fresh_tensors(tmp_path):
    model = Tiny()
    path = saved(tmp_path, model, drop=("fresh.weight", "fresh.bias"))
    report = verify_loaded_weights(model, path, fresh_prefixes=("fresh.",))
    assert report == {"file": str(path), "tensors": 4, "fresh": 2}


def test_a_model_that_skipped_the_load_is_caught(tmp_path):
    """pi0.5's failure: a model returned without its weights."""
    path = saved(tmp_path, Tiny())
    with pytest.raises(StageLoadError, match="differ from the file"):
        verify_loaded_weights(Tiny(), path)


def test_a_key_the_file_lacks_is_caught_unless_declared_fresh(tmp_path):
    model = Tiny()
    path = saved(tmp_path, model, drop=("head.bias",))
    with pytest.raises(StageLoadError, match="does not supply: head.bias"):
        verify_loaded_weights(model, path, fresh_prefixes=("fresh.",))


def test_a_file_key_the_model_lacks_is_caught(tmp_path):
    model = Tiny()
    state = {**model.state_dict(), "stale.weight": torch.zeros(1)}
    save_file({k: v.clone() for k, v in state.items()}, str(tmp_path / "model.safetensors"))
    with pytest.raises(StageLoadError, match="not in the model: stale.weight"):
        verify_loaded_weights(model, tmp_path / "model.safetensors")


def test_tied_weights_written_once_are_not_reported_missing(tmp_path):
    model = Tiny()
    model.head.weight = model.body.weight
    path = saved(tmp_path, model, drop=("head.weight",))
    verify_loaded_weights(model, path)


def test_a_coarser_saved_precision_compares_at_that_precision(tmp_path):
    model = Tiny()
    state = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
    with torch.no_grad():
        for name, value in model.named_parameters():
            value.copy_(state[name].float())
    save_file(state, str(tmp_path / "model.safetensors"))
    verify_loaded_weights(model, tmp_path / "model.safetensors")


def test_only_local_full_weight_directories_are_checked(tmp_path):
    assert local_model_file("lerobot/pi05_base") is None
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")
    assert local_model_file(tmp_path) is None
    (tmp_path / "model.safetensors").write_bytes(b"")
    assert local_model_file(tmp_path) == tmp_path / "model.safetensors"


class SilentLoader:
    """Stands in for pi0.5: 'loads' by returning a freshly built model."""

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls()


class Verified(VerifiedLoadMixin, SilentLoader, Tiny):
    pass


def test_the_mixin_refuses_a_loader_that_returned_an_unloaded_model(tmp_path):
    saved(tmp_path, Tiny())
    with pytest.raises(StageLoadError):
        Verified.from_pretrained(str(tmp_path))


# ---- the real two-stage path, on ACT ---------------------------------------

from lerobot.policies.act.modeling_act import ACTPolicy

from oct_vla.policies.control_act import ControlACTPolicy

from test_control_act_encoder_branches import batch, config, stock_twin


@pytest.mark.parametrize("flags", [{}, {"object_adaln": True}, {"object_incontext": True}])
def test_stage2_loads_every_stage1_tensor_and_adds_only_object_ones(tmp_path, flags):
    torch.manual_seed(0)
    stage1 = stock_twin(ControlACTPolicy(config()))
    stage1.save_pretrained(tmp_path)
    stage2 = ControlACTPolicy.from_pretrained(str(tmp_path), config=config(**flags))
    stage1_keys = set(stage1.state_dict())
    new = [k for k in stage2.state_dict() if k not in stage1_keys]
    assert new and all(k.startswith("model.object_conditioning.") for k in new)
    for key, value in stage1.state_dict().items():
        assert torch.equal(stage2.state_dict()[key], value), key


def test_stage2_layerwise_is_exactly_stage1_after_a_real_load(tmp_path):
    torch.manual_seed(0)
    stage1 = stock_twin(ControlACTPolicy(config()))
    stage1.save_pretrained(tmp_path)
    stage2 = ControlACTPolicy.from_pretrained(str(tmp_path), config=config())
    reloaded = ACTPolicy.from_pretrained(str(tmp_path))
    data = batch()
    torch.testing.assert_close(
        stage2.predict_action_chunk(data), reloaded.predict_action_chunk(data), rtol=0, atol=0
    )


# ---- slim checkpoints -------------------------------------------------------

from oct_vla.policies.stage_loading import SLIM_MARKER, SlimCheckpointMixin, rebuilt_prefixes


class _Config:
    base_model_path = "base"

    def _save_pretrained(self, directory):
        (directory / "config.json").write_text("{}")


class Base(nn.Module):
    """A 'frozen backbone' rebuilt identically on construction, and a head."""

    def __init__(self):
        super().__init__()
        self.config = _Config()
        torch.manual_seed(7)
        self.backbone = nn.Linear(3, 3)
        self.backbone.requires_grad_(False)
        torch.manual_seed(int(torch.randint(0, 10**6, ())))
        self.head = nn.Linear(3, 3)

    def _save_pretrained(self, directory):
        raise AssertionError("slim save must not fall through to the full save")


class Slim(SlimCheckpointMixin, Base):
    rebuilt_prefixes = ("backbone.",)


def test_a_slim_save_leaves_out_only_the_rebuilt_prefix(tmp_path):
    from safetensors import safe_open

    model = Slim()
    model._save_pretrained(tmp_path)
    with safe_open(str(tmp_path / "model.safetensors"), "pt") as handle:
        assert set(handle.keys()) == {"head.weight", "head.bias"}
    assert rebuilt_prefixes(tmp_path) == ("backbone.",)


def test_a_slim_save_refuses_to_drop_a_trainable_parameter(tmp_path):
    model = Slim()
    model.backbone.weight.requires_grad_(True)
    with pytest.raises(StageLoadError, match="trainable"):
        model._save_pretrained(tmp_path)


def test_a_slim_checkpoint_verifies_only_with_its_marker(tmp_path):
    saved_model = Slim()
    saved_model._save_pretrained(tmp_path)
    loaded = Slim()
    loaded.load_state_dict(saved_model.state_dict())
    verify_loaded_weights(loaded, tmp_path / "model.safetensors",
                          fresh_prefixes=rebuilt_prefixes(tmp_path))
    (tmp_path / SLIM_MARKER).unlink()
    with pytest.raises(StageLoadError, match="does not supply: backbone"):
        verify_loaded_weights(loaded, tmp_path / "model.safetensors",
                              fresh_prefixes=rebuilt_prefixes(tmp_path))
