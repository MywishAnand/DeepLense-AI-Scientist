# Fact sheet — ML4PS workshop paper (AI-scientist closed loop)

**Scope:** pipeline agents + ReAct planner + tree architecture search ONLY (the V2
code-gen/data-simulation agent is out of scope). Every number below is sourced from
this repo (branch `feat/architecture-search`); the source path follows each claim.
Items that cannot be verified from the repo are marked **UNVERIFIED**.

---

## A. Pipeline architecture

### A1. Agents and typed contracts

Pipeline (WORKFLOW_DESIGN.md order): data simulation → model design → train → infer
→ analysis → experiment planner. `docs/WORKFLOW_DESIGN.md`

| Contract | Key fields | Source |
|---|---|---|
| `SimConfig` | substructure_type (no_sub/cdm/vortex), model_config_name (Model_I/II), num_images, halo_mass, z_halo, z_source, axion_mass, vortex_mass, cosmology(H0,Om0,Ob0); cross-field physics validators | `src/dlens/schemas/_simulation.py` |
| `SimOutput` | run_id, config, num_generated, image_shape, pixel_value_range, timestamp, output_dir, filenames, backend | `src/dlens/schemas/_simulation.py` |
| `ArchitectureSpec` | name, family (resnet/cnn buildable; vit/equivariant defined but NOT buildable), input_shape, channels, num_classes, **depths** (2–4 stages, 1–6 each), **widths** (8–1024, same length), physics_informed, rationale; `is_buildable()` | `src/dlens/schemas/_model_design.py` |
| `TrainingConfig` | loss, optimizer, learning_rate, batch_size, epochs, weight_decay, lr_scheduler, **dropout**, **augment**, **early_stop_patience** | `src/dlens/schemas/_model_design.py` |
| `TrainResult` | run_id, weights_path, backend, num_classes, metrics{train_accuracy, final_loss}, epochs_run | `src/dlens/schemas/_downstream.py` |
| `InferResult` | run_id, num_samples, num_classes, predictions, probabilities, true_labels, accuracy, backend | `src/dlens/schemas/_downstream.py` |
| `AnalysisResult` | accuracy, macro_auc (one-vs-rest, rank statistic), confusion_matrix, per_class{precision,recall,f1,support}, summary | `src/dlens/schemas/_downstream.py`, computed in `src/dlens/tools/_analysis.py` |
| `ExperimentRun` / `ExperimentState` | per-iteration bundle (architecture, training_config, train/infer/analysis results, planner_decision); state = hypothesis + iteration counters + runs list | `src/dlens/schemas/_experiment.py` |
| `PlannerDecision` | action (train / design_model / simulate / report / stop), rationale, updated_params | `src/dlens/schemas/_experiment.py` |

Note: agent-to-agent payloads are compact — the analysis tool reads the full
`InferResult` from deps rather than LLM-transcribed arguments; the planner receives a
state view with prediction arrays excluded. `src/dlens/tools/_analysis.py`
(AnalysisDeps docstring), `src/dlens/agents/_experiment_planner.py` (`decide()`).

### A2. Planner (ReAct closed loop)

- **Design:** observe latest run's metrics → LLM decides ONE typed change → apply →
  retrain → re-score. Inference is never a decision point.
  `src/dlens/agents/_experiment_loop.py` (module docstring).
- **Metrics received per iteration:** train_accuracy, final_loss (+ holdout_accuracy
  when early stopping on), val accuracy, macro_auc, confusion_matrix, per-class
  P/R/F1 — prediction arrays excluded. `src/dlens/prompts/_planner.py`,
  `src/dlens/agents/_experiment_planner.py`.
- **Decision space (typed knobs):** architecture: name, family, physics_informed;
  training: loss, optimizer, learning_rate, batch_size, epochs, weight_decay,
  lr_scheduler, dropout, augment, early_stop_patience.
  `src/dlens/agents/_experiment_loop.py` `_ARCH_KEYS`/`_CFG_KEYS` (lines 38–42).
- **Stop conditions:** planner report/stop; code-side guards — gap ≤ threshold
  (default 0.06), no val improvement (min_val_delta 0.005) for patience=2
  iterations, iteration budget (default max_iterations=4). Guards recorded as
  `[loop guard]` decisions. `src/dlens/agents/_experiment_loop.py` (`run()` signature
  + guard block).
- **Swappable strategy:** `PlannerStrategy` protocol (`async decide(state) ->
  PlannerDecision`); default `ReActPlannerStrategy`; tree search can implement the
  same protocol. `src/dlens/agents/_experiment_loop.py`,
  `src/dlens/agents/_experiment_planner.py`.

### A3. Tree-based architecture search

- **Generate:** LLM proposes N candidates (default `num_candidates=10`) as
  structured ArchitectureSpecs, constrained to the buildable space; unbuildable
  candidates dropped code-side. `src/dlens/agents/_architecture_search.py`.
- **Judge:** LLM-as-judge ranks all candidates from specs + task (pre-training);
  top-k kept (default `top_k=4`). Same file.
- **Run (fidelity):** each top-k candidate trained for REAL, short —
  `candidate_epochs` (class default 4; **the reported run used 10**) on the 1,800-image
  train subset. Same file + `docs/ARCHITECTURE_SEARCH_RESULTS.md`.
- **Prune:** winner by REAL validation accuracy, code-side (`_best`), not the LLM.
- **Refine:** optional one round (default `rounds=2`, `refine_variants=3`): variants
  of the winner, deduped against already-evaluated (family, depths, widths), trained
  short, pruned again.
- **Handoff:** winner ArchitectureSpec becomes the initial architecture of the
  existing ReAct tuning loop. `scripts/run_ai_scientist.py` (Phase A → Phase B).

### A4. Bias controls (verbatim prompt lines)

- Planner: *"Base decisions purely on the evidence; do not assume any preferred
  architecture family or known-best solution."* `src/dlens/prompts/_planner.py:39`.
- Generator: *"…smaller datasets often punish very large models, but explore the
  space rather than assuming one answer."* `src/dlens/prompts/_arch_search.py:26`.
- Judge: *"…judge purely from the specs and task numbers given; do not assume any
  family is inherently best."* `src/dlens/prompts/_arch_search.py:38`.
- Task description (combined run): *"…judge capacity on the numbers, not by
  assumption."* `scripts/run_ai_scientist.py:72`.
- No lensing-specific or equivariance hints appear in any planner/search prompt
  (checked: `src/dlens/prompts/_planner.py`, `src/dlens/prompts/_arch_search.py`).

### A5. LLM

- Default `gpt-5.6-luna` via the **OpenAI Responses API** (gpt-5.6* rejects function
  tools on chat completions with reasoning enabled); `gpt-5.2` switchable via
  `DLENS_LLM`. `src/dlens/config.py` (`DEFAULT_LLM`, `build_llm()`, lines ~113–130).
- Planner-only trajectory (§C1) was run with **gpt-5.2**; the combined search+tuning
  run (§C2) with **gpt-5.6-luna**. `docs/EXPERIMENT_PLANNER_RESULTS.md`,
  `docs/ai_scientist_run.json` ("llm" field).

## B. Dataset

- **Self-generated with the DeepLenseSim Model_I recipes — NOT the official released
  dataset** (the official Google Drive files are quota-blocked for programmatic
  download). `docs/CLASSIFICATION_RESULTS.md` (Dataset section).
- Generation mirrors `DeepLenseSim/Model_I/sim_{no_sub,cdm,axion}.py` exactly:
  `DeepLens()`; `make_single_halo(1e12)`; substructure `make_no_sub()` /
  `make_old_cdm()` / `make_vortex(3e10)`; `make_source_light()`; `simple_sim()`;
  axion mass sampled 10^U(−24,−22) per image. `scripts/gen_model1_dataset.py`
  (docstring + CLASS_SETUPS + make_lens).
- Stack: lenstronomy 1.9.2 + pyHalo (era-pinned commit 64582db) + colossus, run in
  the Docker image. `docs/CLASSIFICATION_RESULTS.md`; image definition:
  `sandbox/Dockerfile` on branch `feat/simulation-codegen-agent`.
- **Classes (exact names):** `no_sub`, `cdm`, `axion`. `scripts/gen_model1_dataset.py`.
- **Counts:** 3,000/class = 9,000 images, 150×150 px, single channel, float32,
  ~809 MB; split 80/20 seeded → **7,200 train / 1,800 val**; planner/search runs used
  a **1,800-image train subset (600/class)** (`train_small`).
  `docs/CLASSIFICATION_RESULTS.md`, `scripts/prepare_deeplense_dataset.py`,
  `docs/EXPERIMENT_PLANNER_RESULTS.md`; verified on disk: 7200/1800/1800 files.
- **Augmentation:** none in the dataset itself; flips/90° rotations exist only as a
  training-time knob the planner can enable. `src/dlens/tools/_torch_backends.py`
  (`_augment`), `src/dlens/schemas/_model_design.py` (`augment`).

## C. Results (all real runs)

### C0. DEFINITIVE PAPER RUN (2026-07-27) — use THESE numbers for headline claims

Source: `docs/PAPER_RUN_RESULTS.md` + `docs/paper_run.json`. gpt-5.2 for every LLM
role (asserted at run start); held-out test set (1,800 fresh images, seeds
777/778/779, zero MD5 overlap with train/val) evaluated exactly once after all
selection; selection on val only; seeded training; 21.0 min total.

- **Search (Phase A, 7.4 min):** 10 candidates (0.024M–31.5M params, 0 unbuildable);
  judge top-4; 10-epoch real training; winner **cnn_medium_s4** (cnn, depths
  [2,3,3,4], widths [32,64,128,256], 2.54M params, val 0.7733); refinement round: 3
  variants, none better.
- **Tuning arms (identical protocol):** search winner — it0 train 0.9994 / val
  0.8233 (gap 0.1761) → planner `augment` → it1 val **0.8600**, AUC 0.9717.
  resnet34 reference — it0 train 0.9839 / val 0.7517 (gap 0.2322; **identical to the
  Jul 20 run — seeded reproducibility**) → `augment` → it1 val 0.8167, AUC 0.9283.
- **Final val vs TEST:** search winner val 0.8600 / **test 0.8706**, AUC 0.9717 /
  **0.9700**; resnet34 ref val 0.8167 / test 0.8100, AUC 0.9283 / 0.9293.
  **Searched arch beats the reference by +6.1 pts test accuracy with ~8.4× fewer
  parameters; test tracks val for both arms (no val flattery).**
- Per-class TEST (winner): no_sub 0.894/1.000/0.944, cdm 0.885/0.720/0.794, axion
  0.835/0.892/0.862; confusion [600,0,0]/[62,432,106]/[9,56,535]. (Reference
  per-class in `docs/PAPER_RUN_RESULTS.md`.)
- Runtime: A 441.5 s · B1 418.0 s · B2 395.6 s · C 6.1 s · total 1,261.2 s.
- Note: the planner's remedy differed from Jul 20 (augment only vs augment+early-
  stop) — same diagnosis, LLM decision nondeterminism; reported honestly.

### C1. Planner trajectory (overfitting run; gpt-5.2; seeded; subset) — **SUPERSEDED by C0** (kept as the mechanism demo; its it0 numbers are reproduced exactly by C0's reference arm)

Source: `docs/EXPERIMENT_PLANNER_RESULTS.md` + `docs/experiment_loop_trajectory.json`.

| it | Config | Train acc | Val acc | Gap | Val AUC | Decision |
|---|---|---|---|---|---|---|
| 0 | resnet34, no regularization, 25 ep | 0.9839 | 0.7517 | 0.2322 | 0.8922 | `train` → `{augment: true, early_stop_patience: 5}` |
| 1 | + augmentation + early stopping | 0.7926 | 0.8017 | −0.0091 | 0.9187 | `[loop guard]` gap −0.009 ≤ 0.06 → stop |

- Planner iteration-0 rationale (verbatim, truncated): *"Run 0 shows strong
  overfitting: train_accuracy=0.9839 vs val accuracy=0.7517 (gap ~0.23) with no
  augmentation and dropout=0.0. Next smallest intervention… enable data
  augmentation… help the weaker 'cdm' class (recall=0.6267)…"* (JSON `rationale`).
- Deltas: val +5.0 pts (0.7517→0.8017); gap 0.2322→−0.0091; AUC 0.8922→0.9187.
- Wall time 6.4 min. `docs/EXPERIMENT_PLANNER_RESULTS.md`.

### C2. Combined run: tree search + tuning (gpt-5.6-luna; seeded; subset) — **SUPERSEDED by C0** (Luna run; keep only as the model-choice/ablation footnote)

Source: `docs/ARCHITECTURE_SEARCH_RESULTS.md` + `docs/ai_scientist_run.json`.

**Candidates (10 generated, 0 unbuildable; params recomputed from specs):**
cnn_tiny_balanced 0.024M · cnn_small_deep 0.072M · cnn_small_wide 0.093M ·
cnn_medium_multistage 0.294M · cnn_large_morphology 1.958M · resnet18_compact
0.701M · resnet_shallow_wide 1.228M · resnet_deep_narrow 1.094M · resnet_medium_wide
3.166M · resnet_high_capacity 5.934M. (JSON `search.proposed`.)

**Judge top-4** (ranking [3,5,2,7,…]): cnn_medium_multistage, resnet18_compact,
cnn_small_wide, resnet_deep_narrow. (JSON `search.judge_ranking`.)

**Round-1 real results (10 epochs each):** cnn_medium_multistage val 0.5983 /
resnet18_compact 0.5817 / resnet_deep_narrow 0.5794 / cnn_small_wide 0.3572 →
**winner cnn_medium_multistage** (cnn, depths [2,2,2,2], widths [16,32,64,128],
0.294M params). **Refine round:** 3 novel variants (0.4961 / 0.5383 / 0.4333) —
none beat the winner. (JSON `search.rounds`, `search.winner`.)

**Winner's tuning trajectory (Phase B):**

| it | Train acc | Val acc | Gap | AUC | Decision |
|---|---|---|---|---|---|
| 0 | 0.9961 | 0.6294 | 0.3667 | 0.8068 | `train` → `{augment: true}` (overfitting diagnosed) |
| 1 | 0.7756 | 0.7489 | 0.0267 | 0.8996 | `[loop guard]` gap closed → stop |

### C3. Standalone real-training result (full data; agents on gpt-5.2)

Source: `docs/CLASSIFICATION_RESULTS.md`.

- 7,200 train / 1,800 val (full generated dataset); ModelDesignAgent chose
  **resnet34**, lr 3e-4, batch 32, 30 epochs, cross-entropy; training 14.7 min (MPS).
- **Val accuracy 0.8367, macro AUC 0.9429**; train accuracy (final epoch) 0.9971.
- Per class (P / R / F1, n=600 each): no_sub 0.956/0.933/0.944 · cdm
  0.755/0.762/0.758 · axion 0.803/0.815/0.809.
- Confusion (rows=true no_sub,cdm,axion): [560,39,1] / [24,457,119] / [2,109,489].

### C4. Runtime

| Item | Value | Source |
|---|---|---|
| Combined run total | **356.2 s (5.9 min)**: Phase A 187.3 s, Phase B 168.8 s | `docs/ai_scientist_run.json` `runtime_s` |
| Phase-A breakdown | generate 7.1 s · judge 4.9 s · round-1 trainings 22.5/9.5/19.1/11.3 s · refine gen 3.8 s · round-2 trainings 36.9/25.6/46.0 s | JSON `search.timings` |
| Planner-only run | 6.4 min | `docs/EXPERIMENT_PLANNER_RESULTS.md` |
| Full-data resnet34 training | 14.7 min (30 epochs) | `docs/CLASSIFICATION_RESULTS.md` |
| Full-dataset end-to-end | **ESTIMATE ~20–30 min (~4×)** — extrapolation, not measured | `docs/ARCHITECTURE_SEARCH_RESULTS.md` |

### C5. Honest comparisons + ablation findings

- Searched winner tuned: val **0.7489** vs planner-tuned resnet34 on the same subset:
  val **0.8017** — the search currently adds automation/exploration, not accuracy.
  `docs/ARCHITECTURE_SEARCH_RESULTS.md` (Honest findings #3).
- Ablation 1: 4-epoch candidate training → all candidates ≈ chance (~0.35 val);
  ranking was noise. Fixed by 10-epoch evaluations. Same doc (#2).
- Ablation 2: a task line naming "a large ResNet overfit badly" biased the judge to
  tiny models (<0.5M) → underfit winner (final val 0.643). Fixed by neutral wording.
  Same doc (#2).

### C6. Environment

- Apple-Silicon Mac (arm64), macOS 26.5.2; training device **MPS** (CPU fallback).
  Verified from local env; device selection in `src/dlens/tools/_torch_backends.py`.
- torch **2.13.0**; pydantic-ai **2.1.0** (verified from the venv).
- Seeding: `torch.manual_seed(0)` at every training call + `np.default_rng(0)` for
  the early-stop split; dataset split seeded. `src/dlens/tools/_torch_backends.py`
  (lines ~168, ~176), `scripts/prepare_deeplense_dataset.py`.
- LLM temperature: **not set** in our code (checked: no `temperature` anywhere in
  `src/`/`scripts/`) → provider default. Effective server-side default for
  gpt-5.6-luna / gpt-5.2: **UNVERIFIED**.

## D. Reproducibility + gaps (do not overclaim)

1. **Single seed, single run** per experiment — no multi-seed error bars anywhere.
2. **LLM nondeterminism**: temperature unpinned; planner/search decisions may vary
   across reruns even with seeded training.
3. ~~No held-out test set~~ **RESOLVED in the definitive run (C0)**: a fresh
   1,800-image test set (new seed lineage, zero MD5 overlap) is evaluated exactly
   once after all selection; selection happens on val only. (Still true for the
   superseded C1/C2 runs.)
4. Subset (1,800-train) results are **not comparable** to the full-data (7,200) run;
   the paper must not mix them in one table.
5. Dataset is self-generated with the DeepLenseSim recipe, not the released Model_I
   set — distributional equivalence to the official data is assumed, not verified.
6. Short-training fidelity bias (slow-starting large models disadvantaged) is a
   known limitation of the search, documented in
   `docs/ARCHITECTURE_SEARCH_RESULTS.md`.
7. Combined run used Luna; planner-only run used gpt-5.2 — cross-run comparisons
   also change the LLM, not just the strategy.

## E. Related-work hooks (already referenced in our materials)

- **lenstronomy** (Birrer & Amara) + **pyHalo** (Gilman et al.) — cited in
  `docs/CLASSIFICATION_RESULTS.md`; pins in `sandbox/Dockerfile`
  (feat/simulation-codegen-agent branch).
- **DeepLense papers** arXiv:1909.07346, 2008.12731, 2112.12121 — listed in the
  DeepLenseSim README (github.com/mwt5345/DeepLenseSim; local clone
  `~/Personal/GSoC/_research/DeepLenseSim/README.md`).
- **HEPTAPOD** (arXiv:2512.15867) — cited in the Test II notebook / standalone
  lens-agent README (github.com/aatmaj28/deeplense-sim-agent), the design that
  inspired the schema-validated tool interfaces.
- **ML4PS 2025 workshop papers** (HEAL-PINN; super-resolution benchmark; Lens-JEPA;
  FlowLensing) — user-provided PDFs; bibliographic fields **UNVERIFIED** in-repo
  (marked TODO in references.bib).
- ReAct (Yao et al. 2023) and LLM-as-judge (Zheng et al. 2023) — methodological
  anchors for the planner and the search judge (named in
  `docs/EXPERIMENT_PLANNER.md` design discussion as "ReAct").
