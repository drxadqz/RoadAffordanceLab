# Engineering Case Study: C3-FaRNet

This document explains the project as an end-to-end research-engineering system rather than only a model diagram. It is intended for technical reviewers who want to understand the problem decomposition, implementation choices, evaluation discipline, and limitations.

## 1. Problem and constraints

RSCD road-state recognition is a 27-class fine-grained visual task. The hard cases are not conventional object categories with different silhouettes. They often contain the same material under different surface conditions and roughness levels:

```text
water_concrete_slight
wet_concrete_slight
water_concrete_severe
wet_concrete_severe
```

The central engineering difficulty is that several signals are entangled:

- water changes brightness, saturation, specular reflection, and visible texture;
- material identity depends on aggregate, cracks, and directional texture;
- roughness depends on weak local variations that disappear under downsampling;
- the 27 labels have a legal factor structure, but a mistaken early hard route can propagate errors;
- adjacent video frames make experiment protocol and split auditing as important as architecture design.

The system therefore needs both a strong visual representation and an evaluation harness that can expose which physical/factor boundary failed.

## 2. System decomposition

### 2.1 Manifest-driven data layer

All training and evaluation entry points read a common CSV manifest contract. A row records the image path, 27-class label, factor labels, domain identifier, and weak friction-risk interval. This avoids embedding machine-specific directory assumptions in model code.

Relevant code:

- [`src/friction_affordance/datasets/manifest.py`](../src/friction_affordance/datasets/manifest.py)
- [`src/friction_affordance/rscd_factors.py`](../src/friction_affordance/rscd_factors.py)
- [`scripts/build_manifests.py`](../scripts/build_manifests.py)

### 2.2 Visual and physics evidence

The verified S7 model uses a ConvNeXt-Tiny visual carrier with task-specific conditioning modules. Parallel differentiable branches summarize cues such as intensity, saturation, high-frequency energy, glare, dark water, texture erasure, and regional connectedness.

The purpose is not to claim a complete physical renderer. These signals are weak, observable evidence that helps the learned representation distinguish appearance changes from material/roughness changes.

Relevant code:

- [`src/friction_affordance/models/backbone.py`](../src/friction_affordance/models/backbone.py)
- [`src/friction_affordance/models/texture.py`](../src/friction_affordance/models/texture.py)
- [`src/friction_affordance/models/physics_evidence.py`](../src/friction_affordance/models/physics_evidence.py)

### 2.3 Structured factor reasoning

The label parser maps each class to a legal tuple $y=(f,m,r)$. The model can then combine:

- single-factor evidence;
- condition–material, condition–roughness, and material–roughness interactions;
- a full three-factor interaction;
- hard-pair experts for adjacent legal states.

This representation makes an error analyzable: a prediction may have the correct material but the wrong roughness, or the correct roughness but the wrong wet/water state.

Relevant code:

- [`src/friction_affordance/models/c3_farnet.py`](../src/friction_affordance/models/c3_farnet.py)
- [`src/friction_affordance/c3_losses.py`](../src/friction_affordance/c3_losses.py)
- [`src/friction_affordance/rscd_factors.py`](../src/friction_affordance/rscd_factors.py)

### 2.4 Selective optimization and model protection

The formal S7 continuation loads a parent checkpoint and a frozen teacher, then trains about 1.09M selected parameters out of 32.49M total. Anchor consistency, non-focus no-flip constraints, and error-gate supervision reduce regression on already reliable classes.

This is an example of a broader systems principle: when a mature model is strong overall but weak on a narrow boundary, the safest experiment is often an isolated, zero/low-impact update with explicit regression gates—not a simultaneous rewrite of every component.

### 2.5 Evaluation and provenance

The project stores more than a final accuracy number:

- full resolved configuration;
- validation history and training logs;
- 27-class precision/recall/F1/support;
- raw and normalized confusion evidence;
- hard-pair and factor-level metrics;
- per-image predictions and confidence;
- checkpoint size and SHA256;
- historical environment and manifest recovery records.

The public release check rebuilds README plots from these payloads and fails if sample totals, dimensions, links, or headline values drift.

## 3. Measured result and failure analysis

The primary self-contained S7 checkpoint reaches:

| metric | value |
|---|---:|
| Top-1 | 90.6323% |
| Macro-F1 | 88.9197% |
| Weighted F1 | 90.6539% |
| Condition accuracy | 96.5960% |
| Material accuracy | 97.2101% |
| Roughness accuracy | 95.1758% |
| Weakest-class F1 | 75.6931% |

The failure decomposition is as important as the headline result:

- roughness participates in 51.50% of errors;
- condition/friction state participates in 36.34%;
- material participates in 29.78%;
- one mistake may involve more than one factor, so these shares do not sum to 100%;
- `water_concrete_slight` is the weakest class, confirming that water-film appearance and weak roughness are still entangled.

This diagnosis directly motivates the current self-developed backbone research. It also prevents a common failure mode in iterative ML development: adding modules because the global score is low without proving which representation is missing.

## 4. Research-engineering decisions

### Decision A — report Macro-F1 and the weakest class

Top-1 can improve while one difficult class collapses. Model selection and release reporting therefore include equal-class metrics and explicit weakest-class evidence.

### Decision B — preserve exact checkpoint lineage

The formal S7 model is a continuation, not a from-scratch run. Its parent, teacher, resolved config, and lineage are stated explicitly. The slightly higher 90.6404% router record is labelled as an inference variant, not a separately trained checkpoint.

### Decision C — separate verified release from active research

ARCQ/TACT experiments are not placed in the headline table until they beat the frozen public baseline under the same data, initialization, training horizon, and validation procedure. Negative or inconclusive experiments remain research evidence, not marketing results.

### Decision D — make figures executable evidence

README figures are generated by [`scripts/build_public_assets.py`](../scripts/build_public_assets.py), which uses only the Python standard library and reads the committed CSV/JSON results. A reviewer can regenerate every chart without installing the training stack.

## 5. Transferable relevance to multimodal / foundation-model engineering

The repository is a computer-vision project, not an LLM project. Its engineering patterns are nevertheless directly transferable:

- **structured representations:** decomposing a complex output space into interacting latent factors;
- **conditional computation:** applying experts/corrections only when a gate is supported;
- **evaluation design:** separating average performance, tail performance, and mechanism-specific failures;
- **model introspection:** inspecting feature maps, confusion topology, and counterfactual drift;
- **efficient adaptation:** training a small selected parameter set while protecting a mature carrier;
- **experiment infrastructure:** immutable configs, resumable runs, provenance hashes, and fail-fast gates.

These are useful in vision-language models, routing/mixture-of-experts systems, post-training, interpretability, and model-evaluation work.

## 6. Concise resume version

The following wording stays within the verified evidence:

> Built C3-FaRNet, a factor-aware PyTorch/CUDA road-state recognition system that models condition–material–roughness interactions across 27 classes; achieved 90.63% Top-1 and 88.92% Macro-F1 on a frozen 49,500-image RSCD test protocol.

> Developed an auditable ML experiment stack with mixed precision, selective 1.09M/32.49M parameter adaptation, checkpoint lineage and SHA verification, per-class/hard-pair diagnostics, and deterministic result visualization.

> Diagnosed tail failures at the factor level—roughness participated in 51.5% of errors—and designed frozen, same-protocol public-baseline gates for subsequent self-developed backbone research without test-set-driven iteration.

## 7. Honest limitations

- The primary result uses a historical 192×192 letterbox protocol; it should not be compared to a different protocol without rerunning both models.
- The training data contains correlated road frames; group/date-disjoint evaluation remains necessary for a stronger generalization claim.
- Visual labels are not direct tire–road friction measurements.
- The verified S7 carrier is ConvNeXt-Tiny-based; the fully self-developed ARCQ/TACT carrier is active research and is not yet the headline model.
- A single historical continuation seed cannot establish variance; future formal work should report multiple frozen seeds.
