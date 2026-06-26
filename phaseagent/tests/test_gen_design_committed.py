"""Guard the committed generative-design artifact (DoD #4 discrete-diffusion guidance baseline).

`train_generative_design` is a multi-hour reward-guided LoRA fine-tune of DPLM-650M
(a discrete-diffusion PLM) that GENERATES high-function multi-mutants by iterative
masked sampling over the editable positions, evaluated on the complete combinatorial
GB1_Wu assay (every sampled combo is measurable). The artifact `gen_design.json` is
GPU-only and lives on the Modal volume, so this guard activates the moment it is
committed and skips cleanly before then, keeping the tree green while the run bakes.

It asserts (a) structural validity of the committed summary regardless of outcome and
(b) the paper's generative claim: the reward-guided library outscores the unconditioned
DPLM and random libraries while staying under the top-library ceiling.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "outputs" / "epistasis" / "gen_design.json"

# keys the writer (modal_app_v2.py::train_generative_design) always emits
KEYS = [
    "assay", "positions", "epochs",
    "generated_lib_mean_function", "generated_NOVEL_lib_mean_function",
    "n_novel_generated", "coverage",
    "unconditioned_DPLM_mean", "random_lib_mean",
    "top_library_ceiling", "library_mean_overall",
]


def _committed():
    return json.loads(GEN.read_text()) if GEN.exists() else None


def test_gen_design_summary_is_structurally_valid():
    """Always-true sanity on landing: config matches the launched run, ranges valid."""
    s = _committed()
    if s is None:
        pytest.skip("gen_design.json not yet committed (generative run pending)")
    for k in KEYS:
        assert k in s, f"missing key {k}"
    # the launched config: GB1_Wu, 4 editable sites. epochs is the training
    # budget, re-budgeted 5->3 after the iter31 timeout diagnosis (see RUNS.md);
    # assert it is a positive int rather than a magic number so a future
    # re-budgeted relaunch does not turn this guard red on harvest.
    assert s["assay"] == "GB1_Wu"
    assert s["positions"] == [265, 266, 267, 280]
    assert isinstance(s["epochs"], int) and s["epochs"] >= 1
    assert 0.0 <= s["coverage"] <= 1.0
    assert s["n_novel_generated"] >= 0


def test_gen_design_reward_guidance_beats_baselines():
    """The paper's generative claim: reward-guided DPLM generation produces a
    higher-function library than the unconditioned DPLM and random baselines, and
    does not exceed the top-library ceiling (higher DMS_score = better function)."""
    s = _committed()
    if s is None:
        pytest.skip("gen_design.json not yet committed (generative run pending)")
    gen = s["generated_lib_mean_function"]
    assert gen > s["random_lib_mean"], "reward guidance must beat random"
    assert gen > s["unconditioned_DPLM_mean"], "reward guidance must beat unconditioned DPLM"
    assert gen > s["library_mean_overall"], "reward guidance must beat the average combo"
    assert gen <= s["top_library_ceiling"], "cannot exceed the top-library ceiling"
