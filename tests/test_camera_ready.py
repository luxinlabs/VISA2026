import hashlib, json
import pytest
from agentisa import interfaces
from agentisa.pricing import PRICE, trial_cost_usd


def test_trial_cost_discounts_only_reported_cached_tokens():
    log = {"model": "gpt-4.1-mini-2025-04-14", "input_tokens": 1_000_000, "output_tokens": 1_000_000}
    assert trial_cost_usd(log) == pytest.approx(0.40 + 1.60)  # logs without the field: plain input price
    cached = dict(log, cached_input_tokens=600_000)
    assert trial_cost_usd(cached) == pytest.approx(0.4 * 0.40 + 0.6 * 0.10 + 1.60)


def test_every_priced_model_has_input_output_and_cached_rates():
    assert all(len(v) == 3 and v[2] <= v[0] for v in PRICE.values())


def test_full_spec_requires_the_pinned_hash(tmp_path, monkeypatch):
    lock = json.loads((interfaces._ROOT / "data/versions.lock.json").read_text())
    wrong = tmp_path / "spec.md"; wrong.write_text("not the pinned spec")
    monkeypatch.setattr(interfaces, "SPEC_CANDIDATES", (wrong,))
    interfaces.pinned_spec_text.cache_clear()
    with pytest.raises(FileNotFoundError, match=lock["stablehlo_spec_commit"]):
        interfaces.pinned_spec_text()
    good = tmp_path / "good.md"; good.write_bytes(b"pinned")
    monkeypatch.setitem(lock, "stablehlo_spec_sha256", hashlib.sha256(b"pinned").hexdigest())
    monkeypatch.setattr(interfaces, "SPEC_CANDIDATES", (good,))
    monkeypatch.setattr(interfaces.json, "loads", lambda _s, _orig=json.loads: lock)
    interfaces.pinned_spec_text.cache_clear()
    assert interfaces.pinned_spec_text() == "pinned"
    interfaces.pinned_spec_text.cache_clear()
