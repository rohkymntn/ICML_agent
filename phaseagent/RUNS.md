# RUNS — launched Modal runs (story C)

App: `phaseagent-v2` · Volume: `phaseagent-data` (`/data`). Pull artifacts with
`modal volume get phaseagent-data outputs/epistasis/<file> .`

| date | command | app/run id | expected output | status |
|---|---|---|---|---|
| 2026-06-25 | `modal run --detach modal_app_v2.py::train_epistasis_e2e --epochs 10 --lora-r 64 --batch-size 128` | `ap-1MkxsyFTH1crCnu2Osf8QG` | `outputs/epistasis/e2e_results.json` | **FAILED** — died ~4 min after launch (stopped, 0 tasks, no container logs). Root cause: it was building the COLD `gpu_lora_image` (heavy torch/transformers/peft, ~5-10 min) and the iteration-1 client teardown killed it BEFORE the build finished and the function detached. `--detach` only protects a run after the function is enqueued; the foreground image build is not yet detached. |
| 2026-06-25 | `modal run --detach modal_app_v2.py::h100_probe` | `ap-KYORB1zqBYy1ecFYtHXm5E` | (none; returns dict) | **DONE** — succeeded (no prints, so no container logs; app stopped on return). Purpose: pre-warm / cache the `gpu_lora_image` build so the real run starts instantly. Confirmed: subsequent runs build the image in ~3 s (cache hit). |
| 2026-06-25 | `modal run --detach modal_app_v2.py::train_epistasis_e2e --epochs 10 --lora-r 64 --batch-size 128` | `ap-ayiJz1iYjfF0M5qUt0Rhaj` (https://modal.com/apps/yipvincent52/main/ap-ayiJz1iYjfF0M5qUt0Rhaj) | `outputs/epistasis/e2e_results.json` | RUNNING (detached, H100, 10 epochs, warm cache). Verified healthy: container loaded data — `[e2e] 120000 doubles, 149 proteins; train 99927 / 124 prot, holdout 20073 / 25 prot` — and is training. Survives local client kill (still 1 active task after killing the local `modal run` process). |

## Notes
- All training runs MUST be `--detach`, AND the `gpu_lora_image` MUST already be cached, or the run dies when the iteration's local client is torn down mid-build. **Pre-warm the cache with `modal run --detach modal_app_v2.py::h100_probe` first** (or any prior run that finishes the build); then the real run's image build is a ~3 s cache hit, the function enqueues + detaches immediately, and it survives client/iteration teardown (verified).
- The e2e function writes `heldout_protein_median_spearman` (DoD #2) to `e2e_results.json` on the volume and commits the volume. Poll next iteration with `modal volume get phaseagent-data outputs/epistasis/e2e_results.json .`.
- `tectonic` 0.16.9 is now installed locally (`/opt/homebrew/bin/tectonic`) for DoD #1 compilation.
