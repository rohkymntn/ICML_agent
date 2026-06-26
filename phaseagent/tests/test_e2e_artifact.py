"""Guard the committed e2e cross-protein generalization headline (DoD #2).

The end-to-end LoRA fine-tune is a multi-hour H100 run and cannot be recomputed
locally, so the guard asserts the committed summary's keys equal the numbers the
paper cites (section 7.5, ``tab:e2e`` / CLAIMS.md) and pass internal sanity
(held-out protein count, fractions in range, learned transfer beats the zero-shot
reference). This is the no-drift gate between the artifact and the paper.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "epistasis" / "e2e_results.json"


def test_e2e_summary_matches_paper_numbers():
    s = json.loads(RESULTS.read_text())
    # held-out protein evaluation, exactly the DoD #2 ~25 held-out proteins
    assert s["n_holdout_proteins"] == 25
    assert s["n_heldout_prot_scored"] == 25
    assert s["n_proteins"] - s["n_holdout_proteins"] == 124  # training proteins
    assert round(s["heldout_protein_median_spearman"], 2) == 0.33
    assert round(s["heldout_protein_mean_spearman"], 2) == 0.33
    assert round(s["frac_heldout_prot_gt_0p2"], 2) == 0.80
    assert round(s["zero_shot_reference"], 2) == 0.25


def test_e2e_learned_transfer_beats_zero_shot():
    """The paper's claim: cross-protein transfer exceeds the zero-shot reference."""
    s = json.loads(RESULTS.read_text())
    assert s["heldout_protein_median_spearman"] > s["zero_shot_reference"]
    assert 0.0 <= s["frac_heldout_prot_gt_0p2"] <= 1.0
    assert s["trainable_params_M"] > 0
