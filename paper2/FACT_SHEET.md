# Fact sheet — Paper 2: grounded lenstronomy code generation with sandboxed validation

**Scope:** the V2 data-simulation / code-generation agent ONLY (PR #7 lineage:
NL description → version-grounded code-gen → Docker sandbox execution → validation
loop). The AI-scientist loop is out of scope (one sentence of context max).
Sources: branch `feat/simulation-codegen-agent` (system), branch
`exp/model-comparison-codegen` (evaluation), plus git history. Cross-branch paths
are written as `<branch>:<path>`. Unverifiable items marked **UNVERIFIED**.

---

## A. System architecture

### A1. Flow and schemas

NL `SimSpec` → code-gen agent (grounded prompt) → `GeneratedProgram` → sandbox
execution → validation → retry (error fed back) → `CodegenResult`.

| Schema | Fields | Source |
|---|---|---|
| `SimSpec` | description (plain-English physics requirement), notes | `feat/simulation-codegen-agent:src/dlens/schemas/_codegen.py` |
| `GeneratedProgram` | reasoning (inherited from framework OutputSchema), code (self-contained runnable script) | `...:src/dlens/agents/_simulation_codegen.py` |
| `ValidationResult` | passed, checks{ran, produced_output, is_2d, finite, non_trivial}, image_shape, message | `...:src/dlens/schemas/_codegen.py` |
| `CodegenResult` | **reasoning** ("The reasoning process of the agent." — framework convention), spec, code, ok, attempts, validation | same file (reasoning added in commit `ebe7579`) |

- Retry loop: `generate_and_validate()` — up to `max_retries` (default 3) attempts;
  on failure the validation message + stderr (truncated to 2,000 chars) is fed back
  as `last_error`. `...:src/dlens/agents/_simulation_codegen.py`.
- Output convention: the generated script must save its final image to the path in
  the `DLENS_OUTPUT` env var; the sandbox reads it back.
  `...:src/dlens/tools/_sandbox.py` (module docstring).

### A2. Grounding (the paper's core idea)

- A **version-pinned lenstronomy 1.9.2 API cheat-sheet** is appended to the system
  prompt: exact imports, call signatures, and kwargs-as-lists conventions,
  **verified by introspection against the installed version**.
  `feat/simulation-codegen-agent:src/dlens/prompts/_codegen.py`
  (`LENSTRONOMY_API_CHEATSHEET`).
- Representative lines (verbatim):
  - *"LENSTRONOMY API CHEAT-SHEET (verified against the sandbox's installed
    version, lenstronomy 1.9.2 — use these EXACT signatures)"*
  - *"Grid/data (this version uses camelCase numPix/deltaPix):
    `kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, ...)`"*
  - *"PSF (use fwhm; there is NO 'sigma' argument):
    `psf_class = PSF(psf_type='GAUSSIAN', fwhm=0.15, pixel_size=delta_pix)`"*
- Why it exists (documented motivation): LLMs "blend argument names across versions
  (e.g. old camelCase `numPix` vs current snake_case `num_pix`; `sigma` vs `fwhm`
  on PSF)" — same file, module docstring. The casing flip between 1.9.2 and 1.14+
  is documented in the header comment (lines ~14–16).
- **Regeneration mechanism:** `scripts/gen_lenstronomy_cheatsheet.py` introspects
  the *installed* lenstronomy and emits the sheet — run inside the sandbox image
  when the pin changes. Same branch.
- **Era-matched dependency pins** (all verified working): python:3.10-slim, numpy<2,
  scipy<1.14, astropy<6 (newer astropy removes `isiterable`, which lenstronomy 1.9.2
  imports), lenstronomy==1.9.2, pyHalo pinned to 2022-era commit `64582db` and
  installed editable (non-editable drops subpackages like `pyHalo.Cosmology`;
  post-2022-07-10 commits need a newer lenstronomy). `...:sandbox/Dockerfile`
  (comments), commits `4b35054`, `c88954c`.
- **Known gap: pyHalo is NOT covered by the cheat-sheet** — documented as the cause
  of the one evaluation failure (see B1).
  `exp/model-comparison-codegen:docs/MODEL_COMPARISON.md` (Failure modes).

### A3. Sandbox

- **Image** (`...:sandbox/Dockerfile`): python:3.10-slim base; era-pinned stack
  (A2); non-root user `sandbox` (uid 10001); apt upgrade for base CVEs. Live local
  image size **1.58 GB** (verified via `docker images`; includes the editable pyHalo
  clone). Build time: **UNVERIFIED** (not logged).
- **Execution** (`...:src/dlens/tools/_sandbox.py`, `DockerSandbox.run()`): writes
  the script to a temp job dir, mounts it at `/work`, runs
  `--network=none --memory=2g --cpus=1 --pids-limit=256` with a `timeout`-wrapped
  python invocation (default 120 s); output only via the mounted dir
  (`DLENS_OUTPUT=/work/output.npy`).
- **LocalSandbox fallback**: host subprocess, explicitly documented as
  "INSECURE — offline tests / trusted code only". Same file.
- **Verified isolation evidence** (recorded in commit `6b1964b`'s message, run via
  the production `DockerSandbox.run()` path): Linux/linuxkit guest (host: Darwin),
  non-root uid 10001, network blocked (OSError on connect with `--network=none`),
  cgroup caps applied (memory.max = 2 GiB, cpu.max = 1 CPU), lenstronomy 1.9.2
  executed inside.
- Per-run overhead Docker vs LocalSandbox: **UNVERIFIED** (not measured).

### A4. Validation harness

`validate_output()` checks (`src/dlens/agents/_simulation_codegen.py`, branch
`exp/paper2-eval`): ran cleanly (exit 0 + output file produced), output loads as a
numpy array, is 2-D, all values finite, non-trivial (max − min > 0), and — when the
request pins an exact size — `matches_requested_size` (added on this branch, with a
unit test; enforced in the B6 custom_grid category). Validation remains structural,
not scientific (see D). On failure, the retry loop feeds the error back (A1).

### A5. HITL and LLM configuration

- **No HITL gates inside this path** — the generate→validate loop is automatic;
  human review happens downstream ("the validated code is what gets handed to
  Michael", agent module docstring; workflow agreed 2026-07-11, same docstring).
  (The V1 data-simulation agent's clarify/approve gates are a different system.)
- LLM: `DEFAULT_CODEGEN_MODEL = "gpt-5.2"` (flipped from gpt-4o-mini on branch
  `exp/paper2-eval` so the repo matches the paper;
  `src/dlens/agents/_simulation_codegen.py`). All verified runs use **gpt-5.2**. **Luna incompatibility
  finding:** gpt-5.6-* rejects function tools on `/v1/chat/completions` with
  reasoning enabled (HTTP 400: "use /v1/responses or set reasoning_effort to
  'none'"); the eval drives Luna via `OpenAIResponsesModel`.
  `exp/model-comparison-codegen:docs/MODEL_COMPARISON.md`.

## B. Evaluation results (all real)

### B1. 9-prompt model comparison (gpt-5.2 vs gpt-5.6-luna) — model-choice evidence; for grounded-performance headline numbers use B5/B6 (SUPERSEDES the per-arm details here)

Source: `exp/model-comparison-codegen:docs/MODEL_COMPARISON.md` + raw
`docs/model_comparison_results.json` (recomputed — matches). Harness:
`scripts/model_comparison_eval.py` (same branch). 9 synthetic prompts derived from
DeepLenseSim Models I/II/III × {no_sub, cdm, axion}
(`feat/simulation-codegen-agent:src/dlens/data/sim_prompts.py`; Model_IV omitted —
its scripts are empty upstream). Real API calls + real Docker execution; max 3
attempts; identical prompts/settings.

| Metric | gpt-5.2 | gpt-5.6-luna |
|---|---|---|
| Final pass (validated image) | **9/9** | 8/9 |
| First-attempt pass | 2/9 | 3/9 |
| Mean attempts | 1.78 | 1.78 |
| Schema ok / code parses (per generation) | 16/16 · 16/16 | 16/16 · 16/16 |
| Mean generation latency | 13.7 s | 9.4 s |
| Mean wall/prompt (incl. sandbox) | 26.0 s | 18.1 s |
| API | chat completions | requires `/v1/responses` |

**Luna's failure in detail:** `Model_III_axion`, all 3 attempts — generated
`from pyHalo.preset_models import CDM`, which does not exist in the pinned pyHalo
(`ImportError: cannot import name 'CDM' from 'pyHalo.preset_models'`); never
recovered. This import is *outside* the lenstronomy cheat-sheet's coverage — the
same version-grounding gap the sheet fixes for lenstronomy. **Terminology for the
paper: this is version blending, not hallucination** — that import IS correct for
2022-era pyHalo (the DeepLenseSim wrapper itself uses it); the failure occurred
against the image state before the pyHalo era-pin commit `c88954c`.

### B2. End-to-end live verification (in-container)

Commit `6b1964b` (message): `--live` run with gpt-5.2 — generated code executed
**inside the Docker container** via `DockerSandbox` and **passed validation on the
first attempt** (150×150 lensed image, matching the Model_I-style prompt).

### B3. Cheat-sheet before/after evidence

Recorded in git history (commit messages; also in PR #7 comments on GitHub — the
comments themselves are not repo files):
- `2438301`: without grounding, gpt-5.2 "wrote idiomatic code but used old camelCase
  kwargs (numPix) and a nonexistent PSF 'sigma' arg, **failing 3/3 attempts**";
  with the introspection-verified sheet, "same model, same request now **passes
  validation on the FIRST attempt**".
- `4b35054`: sheet re-keyed to 1.9.2 after introspecting it (camelCase confirmed);
  "gpt-5.2 + this cheat-sheet passes validation on the FIRST attempt … generated
  code correctly uses numPix/deltaPix and PSF(fwhm=...)" against a real 1.9.2
  install.
- The anecdotal before/after is now SUPERSEDED by a systematic ablation: see B5
  (`paper2/docs/GROUNDING_ABLATION.md` + raw JSON).

### B4. Scale/robustness data point: 9k-image dataset generation

The same sandbox stack generated the full training dataset used elsewhere: 9,000
Model_I images (3,000/class, 150×150, ~809 MB) — `docs/CLASSIFICATION_RESULTS.md`
(current branch). Zero failed realizations and per-class runtimes (~1,109–1,159 s
for 3,000 images, 3 parallel containers) are recorded only in local generation logs
(`~/Personal/GSoC/deeplense_data/gen_*.log`, outside the repo) — **cite carefully or
regenerate logs; UNVERIFIED-IN-REPO**. The held-out test set (1,800 images) was
also produced by this stack under explicit seeds (docs/PAPER_RUN_RESULTS.md).

### B5. Grounding ablation — FULL 2-arm study (2026-07-27; the paper's core table; supersedes the 9-prompt pilot below)

Source: `paper2/docs/GROUNDING_ABLATION.md`; raw
`paper2/docs/expanded_eval.json` + `paper2/docs/expanded_eval_ungrounded.json`;
harness `scripts/paper2_eval.py` (branch `exp/paper2-eval`). **28 prompts x 3
repeats x 2 arms = 168 runs**, gpt-5.2 both arms, Docker sandbox, 3 attempts; only
difference = cheat-sheet present.

| Metric | Grounded | Ungrounded |
|---|---|---|
| Final pass | **83/84 = 98.8%** [93.6, 99.8] | 70/84 = 83.3% [73.9, 89.8] |
| First-attempt | 60.7% | 51.2% |
| Mean attempts | 1.42 | 1.74 |
| Wall mean / p90 | 14.9 / 27.3 s | 24.6 / 46.7 s |
| Hard failures | 1 | 14 |
| Failed attempts: sig/attr/import/other | 5/15/0/16 | 19/34/2/21 |

CIs do not overlap (+15.5 pts absolute). Version-blending classes cause 55/76
ungrounded failed attempts and 11/14 hard failures; degradation concentrates in
wrapper-recipe prompts (canonical 19/27 vs 26/27) while raw-lenstronomy custom
grids survive (11/12 vs 12/12). Pilot (9 prompts, 1 repeat: 9/9 vs 6/9) kept in
`grounding_ablation.json` — superseded.

### B6. EXPERIMENT B — expanded suite (2026-07-27; headline evaluation)

Source: `paper2/docs/EXPANDED_EVAL.md` + `paper2/docs/expanded_eval.json`. 28 prompts
(9 canonical + 12 param variations + 4 exact-size custom grids [enforced by the new
`matches_requested_size` check] + 3 phrasing styles) × 3 repeats = 84 runs, grounded,
gpt-5.2.

- **Overall 83/84 = 98.8%** (Wilson 95% CI [93.6%, 99.8%]); first-attempt 60.7%;
  mean attempts 1.42; wall mean 14.9 s (p90 27.3 s).
- Per category: canonical 26/27 · param_variation 36/36 · custom_grid **12/12 all
  first-attempt** · phrasing 9/9.
- Single hard failure: `Model_III_axion` rep 2 — the DeepLens wrapper's
  `axion_mass=None` constructor pitfall (deeplense/pyHalo grounding gap, not
  lenstronomy). Only prompt with mixed pass/fail across repeats.
- Ungrounded arm now run across the full expanded suite — see B5 (168-run
  2-arm ablation).

## C. Motivation hooks

- **Manual workflow today:** DeepLenseSim datasets are produced by hand-written
  per-model scripts (`Model_I/sim_{no_sub,cdm,axion}.py` upstream;
  github.com/mwt5345/DeepLenseSim) run by the simulation lead — the agent's NL→code
  path automates authoring new variants. Use case: "the validated code is what gets
  handed to Michael" (agent docstring); intended standalone use by the simulation
  team (`sandbox/README.md`, same branch).
- **Version fragility (the problem grounding solves):** lenstronomy's API changed
  argument casing across versions (numPix→num_pix; PSF sigma vs fwhm); DeepLenseSim
  additionally requires era-matched astropy/scipy/pyHalo — naive codegen against
  "lenstronomy in general" produces plausible-but-broken calls. Sources:
  `prompts/_codegen.py` docstring, `sandbox/Dockerfile` comments, commits
  `4b35054`/`c88954c`, MODEL_COMPARISON failure modes.

## D. Reviewer-facing gaps (do not overclaim)

1. ~~9 prompts, single run~~ **PARTLY RESOLVED (B6)**: now 28 prompts x 3 repeats
   (84 runs) with a Wilson CI. Still synthetic/recipe-derived; no scientist-
   authored prompts yet.
2. **No ground-truth prompts from the simulation team yet** (requested, not yet
   received) and no user study — "does the generated code match what the scientist
   wanted" is untested.
3. **pyHalo/deeplense-wrapper grounding missing** — the one observed hard failure
   is exactly this gap.
4. **Structural, not scientific validation:** checks a finite, non-trivial 2-D
   image; the requested-size check is now implemented (`matches_requested_size`,
   enforced when the prompt pins a size — B6 custom_grid) but physics correctness
   is still not validated.
5. **Single sandbox environment** (one image, one lenstronomy version); portability
   of the grounding claim to other versions is only supported by the regeneration
   script, not by experiments.
6. ~~Before/after grounding is a 1-prompt anecdote~~ **RESOLVED (B5)**: full
   168-run 2-arm ablation over the expanded suite, same model/sandbox/budget;
   non-overlapping CIs.
7. Luna comparison confounds model with API path (chat completions vs Responses).

## E. Related-work hooks

- **HEPTAPOD** (arXiv:2512.15867) — schema-validated tool design inspiration (cited
  in the Test II notebook / standalone repo README; github.com/aatmaj28/deeplense-sim-agent).
- **lenstronomy** (Birrer & Amara 2018), **pyHalo** (Gilman et al. 2020),
  **DeepLense papers** (arXiv:1909.07346, 2008.12731, 2112.12121), **DeepLenseSim**
  (github.com/mwt5345/DeepLenseSim).
- **TODO (need citations, not yet researched):** LLM code generation (e.g. Codex/
  HumanEval line), execution-feedback / self-repair codegen, sandboxed code
  execution for agents, LLM API-hallucination studies, retrieval/documentation-
  grounded codegen. Marked as TODO stubs in references.bib.
