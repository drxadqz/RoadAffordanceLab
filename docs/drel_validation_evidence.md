# DREL D350 validation evidence and claim boundary

> Model: DREL-E B `component_full`. This page reports frozen D350 **validation only**. No development proxy-test or historical 49,500-image formal test was read, parsed, or used to select DREL. Machine-readable values are in [`results/drel_d350_validation/metrics_summary.json`](../results/drel_d350_validation/metrics_summary.json). [中文版](drel_validation_evidence_zh-CN.md).

## 1. Current defensible conclusion

The clean architecture that should advance to later full-RSCD training is DREL-E B `component_full`: bounded-energy directional and radial/reliability ledger evidence, a `0.25 × tanh` write, moment-preserving transitions, retained regional mean/deviation decomposition, constrained matched responses, DropPath 0.10, a 320-D embedding, and a 27-way linear head.

It is the best-supported current **D350 production candidate**. It does not replace the repository's formally tested S7 headline; those two evidence domains remain separate.

## 2. Fair Gate8 three-seed validation

The protocol uses D350 train 9,450 / validation 1,350, native `360×240` tensors, seeds 97/197/297, from-scratch CE-only training, no augmentation, teacher, or pretraining, batch 16 with four-step accumulation, a shared 30-epoch cosine horizon, and independent FP32 validation at epoch 8.

| Seed | Top-1 | Macro-F1 | Bottom-5 | Min-class | WCS F1 | Roughness | Hard pair 1 | Hard pair 2 | Score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 97 | 0.334074 | 0.320235 | 0.172509 | 0.155844 | 0.212766 | 0.568148 | 0.220000 | 0.230000 | 0.296226 |
| 197 | 0.343704 | 0.328804 | 0.186541 | 0.151899 | 0.342342 | 0.611111 | 0.320000 | 0.390000 | 0.306311 |
| 297 | 0.348148 | 0.330983 | 0.173478 | 0.166667 | 0.177215 | 0.580000 | 0.160000 | 0.220000 | 0.306348 |
| **Mean** | **0.341975** | **0.326674** | **0.177509** | **0.158137** | **0.244108** | **0.586420** | **0.233333** | **0.280000** | **0.302962** |
| **Sample std.** | 0.007194 | 0.005681 | 0.007837 | 0.007646 | 0.086911 | 0.022189 | 0.080829 | 0.095394 | 0.005834 |

`Score = 0.4×Top-1 + 0.4×Macro-F1 + 0.2×Bottom-5`. WCS is `water_concrete_slight`. Hard pair 1 is WCS versus `water_concrete_severe`; hard pair 2 is WCS versus `wet_concrete_slight`.

### Frozen RSPNet comparison

The RSPNet envelope is the per-column maximum of RSPNet-M and RSPNet-L under the same D350/native/Gate8 budget. It is a seed97 reference, not a three-seed distribution.

| Metric | DREL Gate8 mean ± std. | RSPNet Gate8 envelope | Mean delta (pp) | Reading |
|---|---:|---:|---:|---|
| Top-1 | **0.341975 ± 0.007194** | 0.322963 | **+1.901** | higher mean |
| Macro-F1 | **0.326674 ± 0.005681** | 0.309301 | **+1.737** | higher mean |
| Bottom-5 | **0.177509 ± 0.007837** | 0.172349 | **+0.516** | higher mean |
| Min-class | **0.158137 ± 0.007646** | 0.126582 | **+3.156** | higher mean |
| WCS F1 | **0.244108 ± 0.086911** | 0.227273 | **+1.684** | higher mean, high seed variance |
| Roughness | 0.586420 ± 0.022189 | **0.608889** | **−2.247** | only metric below envelope |
| Hard pair 1 | **0.233333 ± 0.080829** | 0.190000 | **+4.333** | higher mean, high seed variance |
| Hard pair 2 | **0.280000 ± 0.095394** | 0.260000 | **+2.000** | higher mean, high seed variance |
| Score | **0.302962 ± 0.005834** | 0.280691 | **+2.227** | higher mean |

The defensible statement is: at equal D350 Gate8 validation budget, the three-seed DREL mean is above the frozen single-seed RSPNet-M/L envelope on eight of nine tracked metrics, while roughness is 2.25 percentage points lower.

This is not a formal-test win, a claim that every DREL seed beats every envelope column, or a matched multi-seed significance result.

## 3. Gate30 convergence evidence

| Seed | Top-1 | Macro-F1 | Bottom-5 | Min-class | WCS F1 | Roughness | Score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 97 | 0.385185 | 0.378540 | 0.204044 | 0.144928 | 0.265306 | 0.615556 | 0.346299 |
| 197 | 0.385185 | 0.383431 | 0.252150 | 0.146341 | 0.296875 | 0.627407 | 0.357877 |
| 297 | 0.392593 | 0.386616 | 0.215188 | 0.164384 | 0.339623 | 0.625926 | 0.354721 |
| **Mean** | **0.387654** | **0.382863** | **0.223794** | **0.151884** | **0.300601** | **0.622963** | **0.352966** |
| **Sample std.** | 0.004274 | 0.004057 | 0.025005 | 0.010711 | 0.037233 | 0.006535 | 0.005961 |

Roughness rises from 0.586 at Gate8 to 0.623 at Gate30, showing that the architecture can learn the attribute with more optimization. There is no matched RSPNet Gate30 reference, so 0.623 is convergence evidence and cannot be used as an equal-budget superiority claim over the RSPNet Gate8 value.

## 4. Why regional decomposition remains

The no-regional variant won the early Gate8 screen but failed the preregistered three-seed Gate30 deletion rule:

- per-seed score delta had to be at least +0.002; seed197 was only +0.0005 and seed297 reversed to −0.0070;
- mean Macro-F1 delta had to be at least −0.001; observed delta was −0.002687;
- mean Bottom-5 delta passed at +0.010960.

All hurdles were required, so the frozen conclusion is `REGIONAL_RETAINED`.

## 5. Negative routes excluded from production

RPCC+TERM, all-stage amplitude, late-stage amplitude, BLADE branch separation/selective/adaptive writes, and CORAL roughness supervision failed their frozen Gate3 rules. Regional deletion failed Gate30 confirmation. These negative results prevent unstable, competing, or hindsight-selected mechanisms from accumulating in the production model.

## 6. Remaining limitations

1. Gate8 roughness remains 2.247pp below the RSPNet envelope.
2. Min-class F1 remains low and does not improve monotonically with aggregate metrics.
3. WCS and both hard pairs have substantial seed variance.
4. The RSPNet reference has one seed, so a matched multi-seed significance test is unavailable.
5. DREL has no formal-test result. This is an intentional test firewall, not missing bookkeeping.

An active roughness-targeted experimental route remains outside the production module until it passes its own frozen Gate3/Gate8 decision rule.

## 7. Auditable artifacts

- implementation: [`src/friction_affordance/models/drel.py`](../src/friction_affordance/models/drel.py)
- frozen specification: [`configs/drel/component_full_d350.yaml`](../configs/drel/component_full_d350.yaml)
- tests: [`tests/test_drel.py`](../tests/test_drel.py)
- runnable example: [`examples/drel_quickstart.py`](../examples/drel_quickstart.py)
- machine-readable metrics: [`results/drel_d350_validation/metrics_summary.json`](../results/drel_d350_validation/metrics_summary.json)
- algorithm guide: [DREL architecture](drel_algorithm.md)
