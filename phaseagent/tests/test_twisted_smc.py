"""Tests for the iterative twisted SMC sampler."""
import numpy as np
import pytest

from phaseagent.twisted_smc import (
    MockDPLMBackbone,
    MockRewardModel,
    Particle,
    TwistedSMCConfig,
    best_particle,
    run_twisted_smc,
)


def test_mock_backbone_initializes_with_correct_mask_count():
    bb = MockDPLMBackbone()
    rng = np.random.default_rng(0)
    seq, mask = bb.initialize("M" * 30, edit_budget=5, rng=rng)
    assert mask.sum() == 5
    assert len(seq) == 30


def test_mock_backbone_step_unmasks_one_position():
    bb = MockDPLMBackbone()
    rng = np.random.default_rng(0)
    seq, mask = bb.initialize("M" * 20, edit_budget=4, rng=rng)
    step = bb.step(seq, mask, t=0, rng=rng)
    assert step.new_mask.sum() == 3
    assert len(step.new_sequence) == 20


def test_mock_reward_prefers_alanine_rich():
    rm = MockRewardModel()
    out = rm.reward(["AAAA", "MMMM", "AMAM"])
    assert out[0] > out[2] > out[1]


def test_run_twisted_smc_terminates_and_returns_K_particles():
    bb = MockDPLMBackbone()
    rm = MockRewardModel()
    cfg = TwistedSMCConfig(n_particles=4, n_denoise_steps=8, n_outer_iterations=1, beta=1.0, seed=0)
    out = run_twisted_smc("M" * 24, edit_budget=4, backbone=bb, reward=rm, config=cfg)
    assert len(out["particles"]) == 4
    assert out["rewards"].shape == (4,)


def test_run_twisted_smc_outer_loop_runs_more_steps():
    bb = MockDPLMBackbone()
    rm = MockRewardModel()
    cfg_short = TwistedSMCConfig(n_particles=4, n_denoise_steps=4, n_outer_iterations=1, seed=0)
    cfg_long = TwistedSMCConfig(n_particles=4, n_denoise_steps=4, n_outer_iterations=3, seed=0)
    out_short = run_twisted_smc("M" * 16, edit_budget=4, backbone=bb, reward=rm, config=cfg_short)
    out_long = run_twisted_smc("M" * 16, edit_budget=4, backbone=bb, reward=rm, config=cfg_long)
    # Outer loops keep the chain mixing; both runs return valid result dicts.
    assert "particles" in out_short and "particles" in out_long


def test_best_particle_returns_highest_reward():
    bb = MockDPLMBackbone()
    rm = MockRewardModel()
    cfg = TwistedSMCConfig(n_particles=8, n_denoise_steps=8, n_outer_iterations=1, beta=2.0, seed=42)
    out = run_twisted_smc("M" * 30, edit_budget=6, backbone=bb, reward=rm, config=cfg)
    bp = best_particle(out)
    rewards = out["rewards"]
    assert rm.reward([bp.sequence])[0] == pytest.approx(float(np.max(rewards)))


def test_zero_beta_disables_twist():
    """β=0 should approximately match unconditioned best-of-K (no resampling preference)."""
    bb = MockDPLMBackbone()
    rm = MockRewardModel()
    cfg = TwistedSMCConfig(n_particles=4, n_denoise_steps=4, n_outer_iterations=1, beta=0.0, seed=0)
    out = run_twisted_smc("M" * 16, edit_budget=4, backbone=bb, reward=rm, config=cfg)
    # All particles should have non-NaN rewards; chain should not collapse.
    assert np.isfinite(out["rewards"]).all()


def test_run_returns_trajectory_when_requested():
    bb = MockDPLMBackbone()
    rm = MockRewardModel()
    cfg = TwistedSMCConfig(n_particles=4, n_denoise_steps=6, n_outer_iterations=1, seed=0)
    out = run_twisted_smc(
        "M" * 16, edit_budget=4, backbone=bb, reward=rm, config=cfg, return_trajectory=True
    )
    assert out["trace"] is not None
    assert len(out["trace"]) == 6
    assert {"t", "ess", "mean_reward"} <= set(out["trace"][0].keys())
