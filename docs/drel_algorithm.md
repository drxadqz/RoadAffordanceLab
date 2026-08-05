# DREL-E B `component_full`: a complete, beginner-friendly algorithm guide

> Release date: 2026-08-05. Implementation: [`src/friction_affordance/models/drel.py`](../src/friction_affordance/models/drel.py). Frozen specification: [`configs/drel/component_full_d350.yaml`](../configs/drel/component_full_d350.yaml). [中文版](drel_algorithm_zh-CN.md).

## 1. The idea in plain language

DREL classifies a road image into one of 27 RSCD surface states. Instead of forcing one feature stream to do everything, it separates two responsibilities:

- the **semantic carrier** decides what the road is as a whole;
- the **evidence ledger** preserves fine directional texture, response strength, and reliability, then submits a strictly bounded update to the semantic carrier.

The connection is one-way: the ledger may write evidence into the semantic stream, but semantic features cannot feed back and redefine the ledger. An intuitive analogy is a report with an audit trail: the semantic carrier writes the conclusion, while the ledger can attach controlled evidence but cannot be rewritten by the conclusion.

## 2. Minimal background

A PyTorch image is a `B × C × H × W` tensor: batch size, channels, height, and width. The frozen D350 protocol uses `B × 3 × 360 × 240`. A convolution scans a small learned filter over the image. A stride of two approximately halves height and width. A 1×1 convolution mixes channels at the same position. Batch normalization stabilizes channel scales, and a residual connection learns an update `output = input + update`.

## 3. Frozen architecture

```text
RGB image: B×3×360×240
        │
        ├──────────── one-octave-finer evidence ledger ───────────┐
        │   matched responses → energy/reliability → bounded write│
        │                               │ one-way only             │
        ▼                               ▼                          │
semantic stem → stage 0 → stage 1 → stage 2 → stage 3             │
                   /4        /8        /16       /32               │
                   48ch      96ch      192ch     320ch             │
        │
        ▼
BatchNorm → global average → 320-D embedding → Linear(320, 27)
```

| Stage | Semantic width | Regional blocks | Ledger width | Response kernel | Semantic grid | Ledger response grid |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 48 | 1 | 16 | 7×7 | 90×60 | 180×120 |
| 1 | 96 | 2 | 24 | 5×5 | 45×30 | 90×60 |
| 2 | 192 | 5 | 32 | 3×3 | 23×15 | 45×30 |
| 3 | 320 | 2 | 48 | 3×3 | 12×8 | 23×15 |

The ledger is exactly one octave finer than the corresponding semantic grid. Reflected 3×3 stride-2 averaging maps the ledger evidence onto the semantic grid. The implementation checks this geometry and raises an error instead of silently resizing a mismatch.

## 4. Semantic carrier

### 4.1 Stem

The stem applies a 3×3 stride-2 convolution, BatchNorm, ReLU6, then a depthwise 3×3 stride-2 convolution followed by a 1×1 channel mixer. It maps `3×360×240` to `48×90×60`.

### 4.2 Moment-preserving transition

A conventional stride-2 operation can erase signed texture whose positive and negative responses cancel. DREL transports the local mean and dispersion:

\[
\mu=\operatorname{AvgPool}_{3\times3}(x),\qquad
\sigma=\sqrt{\max(\operatorname{AvgPool}_{3\times3}(x^2)-\mu^2,0)+10^{-6}}.
\]

The concatenated tensor `[μ, σ]` is projected by `Conv1×1 + BatchNorm + ReLU6`. The transition does not claim to reconstruct all source pixels; it ensures that local dispersion cannot disappear merely because its signed mean is close to zero.

### 4.3 Regional contrast block

For normalized input `z`, the block separates a reflected 3×3 local mean `m` from a local deviation `d=z-m`. A 5×5 depthwise convolution reads regional context from `m`, while a dilation-2 3×3 depthwise convolution reads spatial deviation from `d`. Their sum is mixed through a gated 1×1 expansion/compression residual path:

\[
o=\operatorname{ReLU6}(\operatorname{BN}(K^{ctx}_{5\times5}(m)+K^{dev}_{3\times3,d=2}(d))),
\]

\[
y=x+\operatorname{DropPath}\left(\operatorname{BN}
(W_c(\operatorname{ReLU6}(c)\odot\operatorname{sigmoid}(g)))\right).
\]

The `component_no_regional` control looked stronger at Gate8, but failed its preregistered three-seed Gate30 deletion confirmation: seed297 reversed by −0.007 score and the mean Macro-F1 hurdle also failed. The production candidate therefore retains this decomposition.

## 5. Evidence ledger

### 5.1 Constrained matched responses

Each ledger group contains four orientations and two phases, even and odd, so `group_width = 4 × 2 = 8`. One learned canonical even/odd residual is rotated to all four orientations. Every forward pass projects the filters to:

- zero spatial DC per input slice;
- unit filter norm;
- Gram–Schmidt even/odd orthogonality; and
- a residual norm below 0.25 around a fixed, non-degenerate anchor.

For response `r` at group `g`, orientation `o`, and phase `p`, directional energy is

\[
E_{g,o}=r_{g,o,even}^{2}+r_{g,o,odd}^{2}.
\]

Squaring and summing the phase pair makes the measure insensitive to local phase sign while retaining directional texture strength.

### 5.2 Energy and reliability

Total response energy, log-energy, and reliability are

\[
T_g=\sum_o E_{g,o},\qquad
L_g=\frac{1}{2}\log(\max(T_g,10^{-12})),
\]

\[
R_g=\frac{T_g}{T_g+\nu_g},\qquad
\nu_g=10^{-6}+(1-10^{-6})\operatorname{sigmoid}(a_g).
\]

The learned `a_g` sets the group-specific energy scale at which evidence becomes reliable.

### 5.3 Direction and radial coordinates

The retained production direction coordinate uses bounded energy:

\[
B_{g,o}=\frac{E_{g,o}}{1+E_{g,o}},\qquad
D_{g,o}=B_{g,o}-\frac{1}{4}\sum_{o'=1}^{4}B_{g,o'}.
\]

`D` therefore describes relative orientation preference rather than absolute brightness or contrast. The radial coordinate is

\[
A_g=[\tanh(L_g/4),\;2R_g-1].
\]

The two branches answer different questions: direction asks “which way does the texture run?”, while radial evidence asks “how strong is the response and can it be trusted?”.

### 5.4 Strictly bounded one-way write

After deterministic one-octave pooling, independent 1×1 projections map direction and radial evidence to the current semantic width:

\[
u_s=P_s^D(\operatorname{pool}(D_s))+P_s^A(\operatorname{pool}(A_s)),
\]

\[
w_s=0.25\tanh(u_s),\qquad S'_s=S_s+w_s.
\]

Every scalar write is therefore inside `(-0.25, 0.25)`. Both writer projections are initialized to zero, so DREL and the zero-ledger control have bitwise-identical epoch-zero outputs. Evidence enters the semantic carrier only after training supports a non-zero write.

### 5.5 Homogeneous ledger refinement

The first three ledger stages use a lightweight normalized depthwise 3×3 → PReLU → normalized pointwise 1×1 residual block, scaled initially by 0.001. It is bias-free and keeps theorem-bearing parameters and arithmetic in FP32. The final stage uses the matched response directly.

## 6. Classifier and objective

The final 320-channel semantic map is normalized and globally averaged into a 320-D embedding. A single `Linear(320,27)` layer produces the logits. The D350 evidence uses ordinary cross-entropy only:

\[
\mathcal{L}_{CE}=-\log\frac{\exp(z_y)}{\sum_{k=1}^{27}\exp(z_k)}.
\]

There is no pretraining, teacher, augmentation, auxiliary factor loss, or roughness auxiliary loss in the frozen result.

## 7. End-to-end pseudocode

```python
semantic = semantic_stem(image)
ledger = image

for stage in range(4):
    if stage > 0:
        semantic = moment_transition(semantic)

    response = matched_response.at(stage)(ledger)
    decomposition = quotient.at(stage)(response)
    ledger = homogeneous_refiner.at(stage)(response)

    energy = directional_energy(response)
    direction = energy / (1 + energy)
    direction -= mean(direction, over="orientation")
    radial = concat(tanh(log_energy / 4), 2 * reliability - 1)

    raw = direction_writer(pool_one_octave(direction)) \
        + radial_writer(pool_one_octave(radial))
    write = 0.25 * tanh(raw)
    semantic = regional_stage.at(stage)(semantic + write)

embedding = global_average(final_batch_norm(semantic))
logits = linear_27(embedding)
```

## 8. Parameter count and checkpoint schema

- backbone: **2,751,694** trainable parameters;
- linear head: `320×27+27 = 8,667`;
- complete classifier: **2,760,361**.

The release preserves the research `component_full` parameter names and semantic buffers. A local release audit instantiated both versions under the same seed: parameter count, state-dict keys, every state tensor, and forward output were exactly equal.

## 9. Quick start

```python
import torch
from friction_affordance.models.drel import build_drel_component_full

model = build_drel_component_full(num_classes=27, head_init_seed=970027)
image = torch.randn(2, 3, 360, 240)
logits = model(image)
probabilities = logits.softmax(dim=1)
```

To inspect the four ledger stages:

```python
model.eval()
with torch.no_grad():
    result = model(image[:1], return_aux=True)

print(result["drel"]["drel_write_rms"])
print(result["drel"]["drel_mean_reliability"])
```

The runnable example includes inference, cross-entropy, backward, and an optimizer step:

```bash
python examples/drel_quickstart.py
```

## 10. What the production module excludes

The clean module intentionally excludes study routes that failed frozen gates: regional deletion, RPCC/TERM, all-stage and late-stage amplitude, BLADE separation/selective/adaptive writes, and CORAL roughness supervision. An unfinished experimental route is not merged until it passes its preregistered decision rule.

This prevents an “everything we tried” architecture and keeps the public model tied to actual validation evidence.

## 11. Evidence boundary

DREL is the current best-supported **D350 production candidate**, not a replacement for the repository's formally tested S7 headline. At equal Gate8 budget, its three-seed mean exceeds the frozen RSPNet envelope on eight of nine tracked metrics; roughness remains 2.25 percentage points lower. Roughness reaches 0.623 by Gate30, but no equal-budget RSPNet Gate30 reference exists, so that value is convergence evidence rather than a superiority claim.

See [DREL validation evidence](drel_validation_evidence.md) for every reported value, limitation, and allowed claim.
