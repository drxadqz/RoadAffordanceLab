# DREL-QRFME: full-RSCD release

[中文](drel_qrfme_full_release_zh-CN.md) | [Repository home](../README.md) | [Machine-readable evidence](../results/drel_qrfme_epoch097/metrics_summary.json)

## 1. Problem and design hypothesis

RSCD contains 27 coupled road states. A label such as `water_concrete_slight` simultaneously describes friction/condition, material, and roughness. Global appearance helps identify material, but the wet–water and smooth–slight boundaries often depend on weak, directional, scale-dependent responses. A conventional semantic backbone can suppress those responses during downsampling; unconstrained fusion can instead let noisy local evidence dominate.

DREL-QRFME therefore separates **stable semantic context** from an auditable **directional response evidence ledger**, connects them with an explicit norm bound, and reads the ledger through factor-specific semantic queries. The model is one coherent architecture rather than a stack of unrelated attention modules.

## 2. Forward computation

Let the normalized image be \(x\in\mathbb{R}^{3\times224\times224}\). The network produces four semantic maps \(s_l\) and four ledger maps \(e_l\), \(l\in\{1,2,3,4\}\).

### 2.1 Semantic road-context stream

The stem reduces the image by four. Each `RoadContextBlock` normalizes and expands its input, sends half of the channels through a local 3×3 depthwise convolution and the other half through a 7×7 context convolution, projects the concatenation, and applies a small LayerScale residual:

\[
s' = s + \operatorname{DropPath}\left(\gamma\odot W_p\,\sigma\bigl([D_{3\times3}(u),D_{7\times7}(v)]\bigr)\right).
\]

The four semantic widths are `[96, 192, 384, 640]` with depths `[2, 3, 9, 3]`.

### 2.2 Directional Response Evidence Ledger

At every stage, `MatchedResponseConv2d` computes paired directional responses. `EvidenceRefiner` removes local noise without replacing the ledger identity. `RadialCompositionQuotient` decomposes a response group into:

- a signed composition coordinate that preserves directional organization;
- log response energy used to estimate local observability/reliability;
- a bounded radial coordinate.

Reliability is computed after per-sample, per-group normalization:

\[
r_{l,g,p}=\sigma\!\left(\frac{a_{l,g,p}-\mu_{l,g}}{\sqrt{\operatorname{Var}(a_{l,g})+10^{-6}}}\right).
\]

No statistics are shared across samples, preventing cross-sample information leakage.

### 2.3 Bounded evidence writing

The ledger candidate is projected to semantic channels and modulated by reliability. For each sample, `NormBudgetWriter` clips its L2 norm:

\[
\tilde u_l=\tanh(W_c c_l+W_r q_l)\odot r_l,\qquad
u_l=\min\left(1,\frac{b_l\lVert s_l\rVert_2}{\lVert\tilde u_l\rVert_2}\right)\tilde u_l,
\]

\[
s_l\leftarrow s_l+u_l,\qquad \frac{\lVert u_l\rVert_2}{\lVert s_l\rVert_2}\le b_l\le0.05.
\]

Both projection convolutions start at zero, so the evidence path cannot destabilize the initial semantic function. The bound is enforced by computation and logged, not merely encouraged by a penalty.

### 2.4 Query-conditioned RFME

RFME reads stages 2 and 3, which retain spatial detail without the cost/noise of the earliest map or the very coarse final map. For factor axis \(a\in\{\text{friction},\text{material},\text{roughness}\}\), the local descriptor is

\[
z^a_p=W^a_s s_p+W^a_e e_p.
\]

A global semantic query and local key define relevance. Reliability and relevance jointly define a normalized spatial measure:

\[
m^a_p=\operatorname{softmax}_p\left(\log \bar r_p + \frac{\langle q^a,k^a_p\rangle}{\sqrt d}\right),
\qquad \sum_p m^a_p=1.
\]

Descriptors are softly assigned to learned factor codewords \(c^a_k\). Conditional residual moments are

\[
\rho^a_k=\frac{\sum_p m^a_p\,\pi^a_{p,k}(z^a_p-c^a_k)}{\max(\sum_p m^a_p\,\pi^a_{p,k},10^{-6})}.
\]

Stage moments are reduced and multiplicatively conditioned on final semantic features. Factor logits are gathered according to the fixed 27-class factor table and layer-normalized into a class residual \(\Delta\). The final prediction is

\[
y = y_{\mathrm{semantic}} + \tanh(\alpha)\,\Delta.
\]

The gate starts at zero, so training first establishes the semantic baseline and then learns whether the factor residual is useful.

## 3. Theory-to-code contract

| Mathematical role | Implementation |
|---|---|
| Semantic carrier | `src/drel_qrfme/models/drel_qrfme_model.py::RoadContextBlock` |
| Directional ledger | `MatchedResponseConv2d`, `EvidenceRefiner`, `RadialCompositionQuotient` |
| Hard write constraint | `NormBudgetWriter` |
| Query-conditioned measure and codeword residual | `FactorMeasureStage` |
| Fixed factor-to-class mapping | `src/drel_qrfme/rscd_label_factors.py` |
| Final model | `DRELRTSurfaceClassifier` |
| Training-only class-prior correction | Balanced Softmax in `training_engine.py` |

The released checkpoint expects architecture version `drel_rt_rfme_v1`; changing widths, depths, ledger layout, class order, or forward ordering invalidates strict checkpoint compatibility.

## 4. Training protocol

- train / validation / official test: 958,941 / 19,860 / 49,500 images;
- seed 97, 100 full epochs, natural sampling;
- AdamW, learning rate 5e-4, five-epoch warm-up, cosine decay to 1e-6;
- batch size 64 with gradient accumulation 2;
- BF16 training, FP32 validation/evaluation;
- Balanced Softmax during training only;
- scratch initialization, no ImageNet weights, teacher, logit patch, TTA, or ensemble.

The checkpoint score is `0.60 Top-1 + 0.30 Macro-F1 + 0.10 Bottom-5 mean F1`. Epoch 97 is the validation-selected checkpoint. Epoch 100 is lower and is not substituted for the best model.

The public YAML retains the frozen manifest hashes. The original full manifest is not redistributed because it embeds dataset paths; users should generate equivalent local manifests and configure the release path remapping without changing class order or split membership.

## 5. Results

### 5.1 Validation-selected checkpoint

| Epoch | Validation Top-1 | Macro-F1 | Bottom-5 F1 | Checkpoint score |
|---:|---:|---:|---:|---:|
| **97** | **91.309%** | **89.623%** | **75.995%** | **89.272%** |

### 5.2 Official test, direct resize 224

| Top-1 | Macro-F1 | Weighted-F1 | Bottom-5 F1 | Weakest-class F1 | NLL | ECE-15 |
|---:|---:|---:|---:|---:|---:|---:|
| **92.265%** | **90.061%** | **92.283%** | **79.359%** | **75.508%** | 0.22025 | 0.02168 |

Factor accuracies are 97.453% friction, 98.152% material, and 95.523% roughness.

### 5.3 Same-protocol RSPNet-L comparison

| Metric | DREL-QRFME | RSPNet-L rerun | Difference |
|---|---:|---:|---:|
| Top-1 | 92.265% | 92.034% | +0.230 pp |
| Macro-F1 | 90.061% | 89.474% | +0.587 pp |
| Bottom-5 F1 | 79.359% | 77.670% | +1.689 pp |
| Weakest-class F1 | 75.508% | 72.814% | +2.693 pp |

### 5.4 Multi-protocol audit

The same frozen checkpoint evaluated with “resize short side by 1.14 then center crop 224” obtains 92.057% Top-1 and 89.819% Macro-F1. Both protocol payloads are released so readers can separate model behavior from image preprocessing.

## 6. Claim boundary and remaining experiments

The released result is a real full-test result, not a 100k proxy or validation-only score. It supports the claim that DREL-QRFME exceeds the local released RSPNet-L rerun under the same direct-resize protocol. It does **not** yet establish a statistically significant global SOTA because only seed 97 is complete. A paper-ready claim additionally requires multiple seeds, training ablations for DREL/writer/RFME, and direct reproduction of every external comparator under a documented common protocol.

The validation/test metrics show that the largest remaining weakness is `water_concrete_slight`; this should motivate controlled factor-boundary analysis rather than a hand-coded class patch.
