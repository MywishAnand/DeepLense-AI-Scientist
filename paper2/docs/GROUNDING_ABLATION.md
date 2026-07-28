# Grounding ablation — full 2-arm study (168 runs)

**Date: 2026-07-27** · Harness: `scripts/paper2_eval.py` · Raw:
`paper2/docs/expanded_eval.json` (grounded) + `paper2/docs/expanded_eval_ungrounded.json`
**Setup:** the 28-prompt expanded suite × 3 repeats × 2 arms = **168 runs**, gpt-5.2
in both arms, Docker sandbox, max 3 attempts. The ONLY difference between arms: the
version-pinned lenstronomy 1.9.2 cheat-sheet is present (grounded) or stripped
(ungrounded) from the system prompt — task, schema, retry budget identical.

## Headline (per arm, 84 runs each)

| Metric | Grounded | Ungrounded |
|---|---|---|
| **Final pass** | **83/84 = 98.8%** (Wilson 95% [93.6, 99.8]) | 70/84 = 83.3% (Wilson 95% [73.9, 89.8]) |
| First-attempt pass | 51/84 = 60.7% | 43/84 = 51.2% |
| Mean attempts | 1.42 | 1.74 |
| Wall per prompt (mean / p90) | **14.9 s / 27.3 s** | 24.6 s / 46.7 s |
| Hard failures (budget exhausted) | 1 | 14 |

**The confidence intervals do not overlap** — grounding's effect is statistically
clear at this sample size (+15.5 points absolute pass rate).

## Per category (pass / runs)

| Category | Grounded | Ungrounded |
|---|---|---|
| canonical (wrapper recipes, 9×3) | 26/27 | **19/27** |
| parameter variations (12×3) | **36/36** | 31/36 |
| custom grid, raw lenstronomy, size enforced (4×3) | **12/12** | 11/12 |
| phrasing styles (3×3) | 9/9 | 9/9 |

Reading: the ungrounded arm degrades most on the **canonical wrapper-recipe
prompts** (19/27) and parameter variations — the regimes that exercise
version-specific DeepLens/lenstronomy-1.9.2 calls — while generic raw-lenstronomy
usage survives almost intact (11/12). **Grounding matters precisely where
version-pinned APIs are involved.**

## Failure taxonomy (failed attempts, incl. later-recovered)

| Category | Grounded | Ungrounded |
|---|---|---|
| wrong_signature | 5 | 19 |
| wrong_api_attribute | 15 | 34 |
| wrong_api_import | 0 | 2 |
| other (runtime/setup) | 16 | 21 |
| **total failed attempts** | **36** | **76** |

Version-blending classes (signature + attribute + import) account for **55 of the
ungrounded arm's 76 failed attempts** and **11 of its 14 hard failures**
(7 wrong_api_attribute + 4 wrong_signature). The grounded arm's single hard
failure is the DeepLens wrapper's `axion_mass=None` constructor pitfall — a
deeplense/pyHalo grounding gap, outside the lenstronomy sheet's coverage.

## Pilot (superseded): 9-prompt single-repeat ablation

The initial pilot (same design, 9 canonical prompts, 1 repeat/arm) showed the same
pattern — grounded 9/9 vs ungrounded 6/9, version-blending errors causing all
unrecovered failures; raw JSON in `paper2/docs/grounding_ablation.json`. The full
2-arm study above supersedes it for all quantitative claims.

## Notes

- Repeats use fresh LLM sampling (temperature unpinned); training is not involved.
- Terminology: **version blending**, not "hallucination" — many "wrong" calls are
  valid in other lenstronomy/pyHalo versions.
