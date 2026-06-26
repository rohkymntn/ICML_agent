# RUNS — launched Modal runs (story C)

App: `phaseagent-v2` · Volume: `phaseagent-data` (`/data`). Pull artifacts with
`modal volume get phaseagent-data outputs/epistasis/<file> .`

| date | command | app/run id | expected output | status |
|---|---|---|---|---|
| 2026-06-25 | `modal run --detach modal_app_v2.py::train_epistasis_e2e --epochs 10 --lora-r 64 --batch-size 128` | `ap-1MkxsyFTH1crCnu2Osf8QG` (https://modal.com/apps/yipvincent52/main/ap-1MkxsyFTH1crCnu2Osf8QG) | `outputs/epistasis/e2e_results.json` | RUNNING (detached, H100, 10 epochs) |

## Notes
- All training runs MUST be `--detach` (non-detached runs die when the local client is cleaned up).
- The e2e function writes `heldout_protein_median_spearman` (DoD #2) to `e2e_results.json` on the volume and commits the volume.
