# Research Status and Claim Boundary

This page separates completed evidence from active hypotheses. It exists so that repository visitors can tell exactly which statements are supported by a frozen evaluation and which are still under development.

## Status ledger

| component / result | status | public claim |
|---|---|---|
| C3-FaRNet S7 self-contained checkpoint | **verified** | 90.6323% Top-1 and 88.9197% Macro-F1 on the 49,500-image historical RSCD test protocol |
| Parent checkpoint + source-reliable router | **verified inference variant** | 90.6404% Top-1; not a separately trained checkpoint |
| S7 checkpoint lineage and recovery archive | **audited** | parent, teacher, configs, environment records, hashes, and result payloads are indexed |
| ARCQ-Road radial/compositional backbone | **research baseline** | mechanism code and validation work exist; no public SOTA claim |
| TACT-Road tangent/level-control backbone | **paused experiment** | preflight and matched-control protocol passed; accuracy gate not yet completed |
| Cross-protocol comparison with RSPNet | **not established** | must be rerun under identical data, preprocessing, initialization, epoch budget, and selection rule |

## Promotion rule for a new headline model

A candidate cannot replace S7 or be described as better than a public baseline unless all of the following are frozen before test access:

1. identical train/validation/test manifests;
2. identical input geometry and preprocessing;
3. identical pretrained/scratch status;
4. identical seed set, optimizer, epoch horizon, and effective batch size;
5. independent FP32 validation of one uniquely selected checkpoint per model;
6. configuration, source, analysis-script, and checkpoint SHA receipts;
7. improvement in Top-1 and Macro-F1 without unacceptable weakest-class regression;
8. one final test evaluation after the model and decision rule are frozen.

Development proxy-test and historical formal test results are not used for architecture selection.

## Current measured bottleneck

The frozen S7 result contains 4,637 errors. Factor attribution reports:

| factor involved in error | share of all errors |
|---|---:|
| roughness | 51.50% |
| condition / friction state | 36.34% |
| material | 29.78% |

Shares overlap because one 27-class error can change more than one factor. The weakest class is `water_concrete_slight` (75.6931% F1), and the lowest hard boundaries combine wet/water appearance with concrete slight/severe roughness.

## Active design direction

The current self-developed-backbone work tests a specific hypothesis: road recognition needs both stable regional organization and weak texture evidence. Earlier ARCQ experiments preserved many local responses but produced spatially fragmented late feature maps. TACT-Road therefore introduces a matched treatment/control experiment that asks whether central-versus-neighborhood tangent evidence improves region organization beyond an equal-parameter symmetric level control.

The experiment was paused only to prepare this public repository. Its process state and recovery checkpoints are preserved outside the public release worktree. No accuracy conclusion is claimed before the frozen validation gate completes.

## What is intentionally not claimed

- “The current repository is the RSCD SOTA.”
- “ARCQ already outperforms RSPNet.”
- “The model directly measures physical tire–road friction.”
- “The source-reliable router is a separately trained S7 model.”
- “A validation-only or small-subset result proves full-dataset superiority.”
