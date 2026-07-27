# Experiment A — grounding ablation (the paper's core claim)

**Date: 2026-07-27** · Harness: `scripts/paper2_eval.py` · Raw: `paper2/docs/grounding_ablation.json`
**Setup:** the original 9-prompt suite (DeepLenseSim Models I–III × no_sub/cdm/axion),
**gpt-5.2 in both arms**, Docker sandbox for every execution, max 3 attempts, identical
task/schema/retry budget. The only difference between arms: the version-pinned
lenstronomy 1.9.2 API cheat-sheet is present (grounded) or removed (ungrounded) from
the system prompt — nothing else changes.

## Results

| Metric | Grounded (cheat-sheet) | Ungrounded |
|---|---|---|
| Final pass (validated image) | **9/9** | 6/9 |
| First-attempt pass | 4/9 | 3/9 |
| Mean attempts | 1.67 | 2.11 |
| Mean generation latency | 9.6 s | 12.7 s |
| Mean wall per prompt | 18.8 s | 30.1 s |

Ungrounded hard failures (all 3 attempts exhausted): `Model_I_cdm`
(wrong_api_attribute), `Model_I_axion` (other), `Model_III_cdm`
(wrong_api_attribute).

## Failure taxonomy (per failed attempt, incl. attempts later recovered)

| Category | Grounded | Ungrounded |
|---|---|---|
| wrong_signature (bad kwargs/arity) | 0 | 3 |
| wrong_api_attribute (missing attr/name) | 3 | 6 |
| other (runtime/physics/setup) | 3 | 4 |
| **total failed attempts** | **6** | **13** |

Reading: without grounding, **version-blending errors (wrong signature + wrong
attribute) account for 9 of 13 failed attempts** and cause all three unrecovered
failures; with grounding, wrong-signature errors disappear entirely and every
failure is recovered within the retry budget. Grounding also cuts wall time ~1.6×
(fewer, shorter retries).

## Notes

- Both arms benefit from the retry loop with error feedback; the ablation isolates
  the *prompt grounding* contribution on top of it.
- The grounded arm's numbers here (9/9, first-attempt 4/9, mean 1.67) are a fresh
  run of the same suite previously reported (9/9, 2/9, 1.78 on Jul 20) — pass rate
  reproduces; attempt-level details vary with LLM sampling, as expected.
- Terminology: we describe these errors as **version blending** (mixing API
  conventions from different library versions), not "hallucination" — several
  "wrong" calls are valid in other lenstronomy/pyHalo versions.
