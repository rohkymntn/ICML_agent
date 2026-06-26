"""Phase-1 validation: do the trained rescue predictor's top-k candidates
recover the 10 curated literature-known intragenic suppressors?

For each (pathogenic, known suppressor) pair, we:
  1. Enumerate every candidate single-AA second-site mutation in the protein
  2. Score every candidate with the trained predictor
  3. Report the rank of the known suppressor among all candidates
  4. Compare to a random baseline (uniform random ranking)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phaseagent.literature_suppressors import LITERATURE_SUPPRESSORS, to_dataframe  # noqa: E402
from phaseagent.rescue_predictor import (  # noqa: E402
    RescuePredictor,
    enumerate_candidate_pairs_for_target,
)


# Approximate WT sequences (truncated when needed). For the rescue scoring we
# only need the residues at m1 and m2 positions; full sequence is used to enumerate
# candidates. These are the canonical UniProt sequences for the curated proteins.
WT_SEQUENCES: dict[str, str] = {
    # P53 (P04637), 393 aa
    "P04637": (
        "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQ"
        "KTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGN"
        "LRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELP"
        "PGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
    ),
    # CFTR (P13569) — 1480 aa, truncated to first 600 for enumeration cost
    "P13569": (
        "MQRSPLEKASVVSKLFFSWTRPILRKGYRQRLELSDIYQIPSVDSADNLSEKLEREWDRELASKKNPKLINALRRCFFWRFMFYGIFLYLGEVTKAVQPL"
        "LLGRIIASYDPDNKEERSIAIYLGIGLCLLFIVRTLLLHPAIFGLHHIGMQMRIAMFSLIYKKTLKLSSRVLDKISIGQLVSLLSNNLNKFDEGLALAHF"
        "VWIAPLQVALLMGLIWELLQASAFCGLGFLIVLALFQAGLGRMMMKYRDQRAGKISERLVITSEMIENIQSVKAYCWEEAMEKMIENLRQTELKLTRKAA"
        "YVRYFNSSAFFFSGFFVVFLSVLPYALIKGIILRKIFTTISFCIVLRMAVTRQFPWAVQTWYDSLGAINKIQDFLQKQEYKTLEYNLTTTEVVMENVTAF"
        "WEEGFGELFEKAKQNNNNRKTSNGDDSLFFSNFSLLGTPVLKDINFKIERGQLLAVAGSTGAGKTSLLMVIMGELEPSEGKIKHSGRISFCSQFSWIMPG"
        "TIKENIIFGVSYDEYRYRSVIKACQLEEDISKFAEKDNIVLGEGGITLSGGQRARISLARAVYKDADLYLLDSPFGYLDVLTEKEIFESCVCKLMANKTR"
    ),
    # lac repressor (P03023), 360 aa
    "P03023": (
        "MKPVTLYDVAEYAGVSYQTVSRVVNQASHVSAKTREKVEAAMAELNYIPNRVAQQLAGKQSLLIGVATSSLALHAPSQIVAAIKSRADQLGASVVVSMVE"
        "RSGVEACKAAVHNLLAQRVSGLIINYPLDDQDAIAVEAACTNVPALFLDVSDQTPINSIIFSHEDGTRLGVEHLVALGHQQIALLAGPLSSVSARLRLAG"
        "WHKYLTRNQIQPIAEREGDWSAMSGFQQTMQMLNEGIVPTAMLVANDQMALGAMRAITESGLRVGADISVVGYDDTEDSSCYIPPLTTIKQDFRLLGQTS"
        "VDRLLQLSQGQAVKGNQLLPVSLVKRKTTLAPNTQTASPRALADSLMQLARQVSRLESGQ"
    ),
    # T4 lysozyme (P00720), 164 aa
    "P00720": (
        "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALI"
        "NMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL"
    ),
    # GFP (P42212), 238 aa
    "P42212": (
        "MASKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIF"
        "FKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNH"
        "YLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"
    ),
    # HIV protease (P12497), short fragment we use the ~99 aa protease region
    "P12497": (
        "PQVTLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
    ),
    # Staphylococcal nuclease (P00644), 149 aa mature form
    "P00644": (
        "MATSTKKLHKEPATLIKAIDGDTVKLMYKGQPMTFRLLLVDTPETKHPKKGVEKYGPEASAFTKKMVENAKKIEVEFDKGQRTDKYGRGLAYIYADGKMV"
        "NEALVRQGLAKVAYVYKPNNTHEQHLRKSEAQAKKEKLNIWSEDNADSGQ"
    ),
    # alpha-1 antitrypsin (P01009), 418 aa
    "P01009": (
        "MPSSVSWGILLLAGLCCLVPVSLAEDPQGDAAQKTDTSHHDQDHPTFNKITPNLAEFAFSLYRQLAHQSNSTNIFFSPVSIATAFAMLSLGTKADTHDEI"
        "LEGLNFNLTEIPEAQIHEGFQELLRTLNQPDSQLQLTTGNGLFLSEGLKLVDKFLEDVKKLYHSEAFTVNFGDTEEAKKQINDYVEKGTQGKIVDLVKEL"
        "DRDTVFALVNYIFFKGKWERPFEVKDTEEEDFHVDQVTTVKVPMMKRLGMFNIQHCKKLSSWVLLMKYLGNATAIFFLPDEGKLQHLENELTHDIITKFL"
        "ENEDRRSASLHLPKLSITGTYDLKSVLGQLGITKVFSNGADLSGVTEEAPLKLSKAVHKAVLTIDEKGTEAAGAMFLEAIPMSIPPEVKFNKPFVFLMIE"
        "QNTKSPLFMGKVVNPTQK"
    ),
}


def main():
    # Load trained predictor.
    with open("/tmp/rescue_predictor.pkl", "rb") as f:
        bundle = pickle.load(f)
    model: RescuePredictor = bundle["model"]
    print(f"loaded predictor trained on {len(bundle['train_proteins'])} proteins")

    suppressor_df = to_dataframe()
    rows = []
    for record in LITERATURE_SUPPRESSORS:
        wt_seq = WT_SEQUENCES.get(record.uniprot)
        if wt_seq is None:
            print(f"[skip] {record.protein}: no WT sequence for {record.uniprot}")
            continue
        path = record.pathogenic
        # Skip non-substitution pathogenics like F508del.
        if "del" in path.lower() or "ins" in path.lower():
            print(f"[skip] {record.protein}: {path} is indel, not handled")
            continue
        try:
            m1_pos = int(path[1:-1])
        except ValueError:
            print(f"[skip] {record.protein}: cannot parse {path}")
            continue
        if m1_pos < 1 or m1_pos > len(wt_seq):
            print(f"[skip] {record.protein}: {path} pos {m1_pos} out of range (len {len(wt_seq)})")
            continue
        wt_aa_at_m1 = wt_seq[m1_pos - 1].upper()
        if wt_aa_at_m1 != path[0].upper():
            print(f"[skip] {record.protein}: {path} WT at pos {m1_pos} is {wt_aa_at_m1!r}, expected {path[0]!r}")
            continue

        # We don't have a measured DMS score for the pathogenic; use a low default
        # ("damaging-but-finite") so the predictor's m1_score feature is in-distribution.
        m1_score_proxy = 0.05

        candidates = enumerate_candidate_pairs_for_target(
            m1_notation=path,
            m1_score=m1_score_proxy,
            seq=wt_seq,
            dataset_id=record.uniprot,
            forbidden_aas=("C", "M", "W", "P"),  # match training-time policy
        )
        if len(candidates) == 0:
            print(f"[skip] {record.protein}: no candidates enumerated")
            continue
        scores = model.predict_rescue_proba(candidates)
        candidates = candidates.copy()
        candidates["rescue_score"] = scores
        candidates = candidates.sort_values("rescue_score", ascending=False).reset_index(drop=True)

        # Find the rank of the literature-known suppressor.
        sup_pos_str = record.suppressor[1:-1]
        if not sup_pos_str.lstrip("-").isdigit():
            print(f"[skip] {record.protein}: cannot parse suppressor {record.suppressor}")
            continue
        sup_pos = int(sup_pos_str)
        sup_aa = record.suppressor[-1].upper()
        match = candidates[(candidates["m2_pos"] == sup_pos) & (candidates["m2_aa"] == sup_aa)]
        if len(match) == 0:
            # Suppressor's mutant residue is in the forbidden set or self-position;
            # rerun without the forbidden filter to record the rank.
            relax = enumerate_candidate_pairs_for_target(
                m1_notation=path,
                m1_score=m1_score_proxy,
                seq=wt_seq,
                dataset_id=record.uniprot,
                forbidden_aas=(),
            )
            relax_scores = model.predict_rescue_proba(relax)
            relax = relax.copy()
            relax["rescue_score"] = relax_scores
            relax = relax.sort_values("rescue_score", ascending=False).reset_index(drop=True)
            match = relax[(relax["m2_pos"] == sup_pos) & (relax["m2_aa"] == sup_aa)]
            n_total = len(relax)
            note = "found_only_after_relaxing_forbidden_filter"
        else:
            n_total = len(candidates)
            note = "found_in_default_pool"
        if len(match) == 0:
            rank = -1
            score = float("nan")
        else:
            rank = int(match.index[0])  # 0-indexed
            score = float(match.iloc[0]["rescue_score"])
        rows.append(
            {
                "protein": record.protein,
                "uniprot": record.uniprot,
                "pathogenic": path,
                "suppressor": record.suppressor,
                "rescue_level": record.rescue_level,
                "rank_0idx": rank,
                "n_candidates": int(n_total),
                "rank_percentile": float(rank / max(n_total - 1, 1)) if rank >= 0 else float("nan"),
                "rescue_score": score,
                "note": note,
            }
        )
        print(f"[ok ] {record.protein:25s} {path:8s} → {record.suppressor:8s} rank {rank+1}/{n_total} (top {100*rank/max(n_total-1,1):.1f}%), score={score:.3f}")

    out_df = pd.DataFrame(rows)
    out_path = ROOT / "outputs" / "clin" / "literature_suppressor_recovery.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    print(f"\nrecovery summary:")
    valid = out_df[out_df["rank_0idx"] >= 0]
    if len(valid) > 0:
        print(f"  evaluated:           {len(valid)}/{len(out_df)} suppressors")
        print(f"  median rank:         {int(valid['rank_0idx'].median())+1} of {int(valid['n_candidates'].median())}")
        print(f"  median percentile:   top {100*valid['rank_percentile'].median():.1f}%")
        print(f"  best:                {valid['note'].iloc[valid['rank_0idx'].idxmin()]}, rank {valid['rank_0idx'].min()+1}")
        random_top1pct = float((valid["rank_percentile"] <= 0.01).mean())
        random_top10pct = float((valid["rank_percentile"] <= 0.10).mean())
        print(f"  in top 1%:           {random_top1pct:.0%}  (random expectation 1%)")
        print(f"  in top 10%:          {random_top10pct:.0%}  (random expectation 10%)")


if __name__ == "__main__":
    main()
