# Experiment B — expanded evaluation suite (grounded, gpt-5.2)

**Date: 2026-07-27** · Harness: `scripts/paper2_eval.py` · Raw: `paper2/docs/expanded_eval.json`
**Setup:** 28 prompts × 3 repeats = **84 runs**, all WITH grounding, gpt-5.2, Docker
sandbox, max 3 attempts. Suite (`src/dlens/data/eval_prompts.py`): the 9 canonical
DeepLenseSim-derived prompts + 12 systematic parameter variations (halo masses
5e11–5e12, lens/source redshift combinations, specific axion/vortex masses,
Euclid/HST instrument variants) + 4 custom-grid/PSF requests via raw lenstronomy
with **exact sizes enforced by the new `matches_requested_size` validation check**
(128², 96², 200², 64²) + 3 phrasing styles (terse / verbose prose / bulleted) of the
same physics. Repeats use fresh LLM sampling (temperature unpinned).

## Headline

| Metric | Value |
|---|---|
| **Overall pass rate** | **83/84 = 98.8%** (Wilson 95% CI [93.6%, 99.8%]) |
| First-attempt pass | 51/84 = 60.7% |
| Mean attempts | 1.42 |
| Wall time per prompt | mean 14.9 s · median 12.8 s · p90 27.3 s · max 33.9 s |

## Per category

| Category | Pass | First-attempt | Mean attempts |
|---|---|---|---|
| canonical (9 × 3) | 26/27 | 13/27 | 1.59 |
| param_variation (12 × 3) | **36/36** | 18/36 | 1.50 |
| custom_grid, exact size enforced (4 × 3) | **12/12** | **12/12** | **1.00** |
| phrasing (3 × 3) | 9/9 | 8/9 | 1.11 |

Notable: the **custom-grid category is perfect on the first attempt in all 12 runs**
— raw-lenstronomy generation with explicit sizes (and the strictest validation) is
the agent's strongest regime, and phrasing style barely matters (9/9 across terse/
verbose/bulleted).

## Failure analysis

- **One hard failure** (all 84 runs): `Model_III_axion`, repeat 2 —
  `TypeError: float() argument must be a string or a real number, not 'NoneType'`.
  This is the DeepLenseSim wrapper's known pitfall: `make_vortex()` requires
  `axion_mass` to be set in the `DeepLens(...)` constructor; the generated code
  omitted it on that repeat. It is a **wrapper-usage (deeplense/pyHalo) grounding
  gap**, not a lenstronomy one — consistent with the cheat-sheet's coverage
  boundary. The same prompt passed in the other 2 repeats (the only prompt with
  mixed pass/fail across repeats).
- Failed-attempt taxonomy across all runs (most recovered by the retry loop):
  other 16 · wrong_api_attribute 15 · wrong_signature 5.

## Relation to Experiment A

Experiment A (the 9-prompt with/without-cheat-sheet ablation, same harness) stands
as the grounding ablation: grounded 9/9 vs ungrounded 6/9, with version-blending
errors causing all unrecovered ungrounded failures
(`paper2/docs/GROUNDING_ABLATION.md`). The ungrounded arm was not repeated across
the expanded suite (time/budget); the paper should state that explicitly.
