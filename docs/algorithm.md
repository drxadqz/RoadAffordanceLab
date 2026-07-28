# C3-FaRNet Algorithm Explanation

## 1. Task

The task is 27-class RSCD road-surface recognition. A class such as `water_concrete_slight` is treated as a coupled state:

```text
water_concrete_slight = friction state water + material concrete + roughness slight
```

This matters because the difficult errors are not random. They often occur on factor boundaries:

- wet vs water
- asphalt vs concrete
- smooth vs slight vs severe
- water-concrete-slight vs wet-concrete-slight

The model is therefore designed around factor coupling, not only flat classification.

## 2. Input

The verified historical run resizes each image to 192 x 192 with letterbox padding. This preprocessing is preserved here because changing it would invalidate exact checkpoint reproduction. Later audits showed that fixed padding can become a shortcut feature, so current from-scratch research uses native 360 x 240 geometry and treats padding removal as a retraining-time change rather than an inference patch.

## 3. Label Factorization

Each label is parsed into three factors:

```text
y = (f, m, r)
```

where:

- `f` is the friction/condition factor: dry, wet, water, fresh_snow, melted_snow, ice
- `m` is the material factor: asphalt, concrete, mud, gravel, none
- `r` is the roughness factor: smooth, slight, severe, none

The code for this factor parsing is in:

```text
src/friction_affordance/rscd_factors.py
```

The factor graph also builds hard pairs. A hard pair is a pair of classes that differs by only one factor, for example:

```text
water_concrete_slight <-> wet_concrete_slight
dry_concrete_smooth <-> dry_concrete_slight
wet_asphalt_smooth <-> wet_asphalt_slight
```

These hard pairs are used to focus correction on real RSCD ambiguities.

## 4. Visual Backbone

The current verified model uses:

```text
convnext_tiny_gate_calibrated_tensor_coupling_concrete_film_rough_stem
```

The instantiated model has about 32.49M parameters. In the verified S7 prefix-tuning setup, only about 1.09M parameters are trainable:

```text
backbone.gate_calibrated_tensor_coupling_banks
pairwise_hardpair_experts
pairwise_hardpair_error_gates
```

This is a ConvNeXt-Tiny-style backbone with task-specific conditioning modules inserted into the feature extractor. The key point is that physical cues are not only appended at the final head. Instead, evidence about water film, concrete texture, roughness and texture erasure can condition earlier feature processing.

Why this is useful:

- late heads can only adjust the final decision
- early/mid feature conditioning can change what texture and boundary information the visual stream preserves
- RSCD errors are often caused by feature entanglement, not only by classifier weights

## 5. PhysicsTexture Branch

The PhysicsTexture branch computes low-level visual evidence related to road friction affordance. It uses differentiable image statistics such as:

- grayscale intensity
- saturation
- local gradient
- Laplacian/high-frequency response
- local contrast
- specular highlight proxy
- dark-water proxy
- snow/ice-like whiteness
- texture-erasure proxy
- soft connectedness of wet/snow regions

Typical internal cues include:

```text
g = 0.299R + 0.587G + 0.114B
```

`g` is grayscale intensity.

```text
s = (max(R,G,B) - min(R,G,B)) / (max(R,G,B) + eps)
```

`s` is color saturation. Low saturation plus high brightness can indicate snow, ice, glare or pale concrete.

```text
E_wet = clip(E_specular + 0.5 E_dark-water, 0, 1)
```

`E_wet` is a wetness proxy. It combines bright specular water highlights and dark smooth puddle-like regions.

```text
E_rough = sigmoid((||grad g|| + 0.5 |lap(g)| - 0.12) * 12)
```

`E_rough` is a roughness proxy. Gradient and Laplacian energy are used because rough road aggregate creates local high-frequency changes.

Implementation:

```text
src/friction_affordance/models/texture.py
PhysicsTextureBranch
```

## 6. LocalPhysicsField Branch

The LocalPhysicsField branch does not reduce the image to only one global vector. It builds local evidence maps for:

- specular water
- dark smooth water
- thin film
- texture loss
- rough gradients
- local contrast

Then it pools these maps over local regions. This is useful because friction evidence may occupy only part of a patch. For example, a thin water film can cover only a small region, while the rest of the patch still looks like dry concrete.

Implementation:

```text
src/friction_affordance/models/texture.py
LocalPhysicsFieldBranch
```

## 7. SemanticPhysicsAttention Branch

SemanticPhysicsAttention builds class-relevant physical evidence instead of treating all cues equally. For example:

- water/wet classes should attend more to mirror water, dark water and thin film
- roughness classes should attend more to rough aggregate and texture fragmentation
- snow/ice classes should attend more to snow-like brightness and low texture

This branch helps the model decide which physical evidence matters for which label family.

Implementation:

```text
src/friction_affordance/models/texture.py
SemanticPhysicsAttentionBranch
```

## 8. Coupled Factor Head

A simple additive model would assume:

```text
score(f,m,r) = A_f + B_m + C_r
```

But RSCD is harder than that. The visual expression of `wet + concrete + slight` is not equal to the sum of wet, concrete and slight features. Therefore, the model uses pairwise and triple interactions:

```text
Z(f,m,r) = A_f + B_m + C_r + D_fm + E_fr + G_mr + H_fmr
```

where:

- `A_f`, `B_m`, `C_r` model single-factor evidence
- `D_fm`, `E_fr`, `G_mr` model two-factor coupling
- `H_fmr` models full three-factor coupling

This is the mathematical reason for using a coupled tensor-style head rather than only a flat linear classifier.

## 9. Hard-Pair Error-Gated Calibration

The current verified S7 run uses:

```text
head_type: hardpair_error_gated_calibrated
```

This head adds correction only around known hard boundaries. The correction is gated so that it does not freely rewrite all classes. In plain language:

1. The base classifier predicts 27 logits.
2. The model checks whether the sample is near a known hard pair boundary.
3. If a hard-pair expert is confident, it applies a small calibrated correction.
4. If the sample is not near that boundary, the correction stays weak.

This design was chosen because many earlier experiments improved one weak class but damaged other classes. A gated hard-pair correction tries to improve weak classes without hurting already stable classes.

## 10. Source-Reliable Boundary Router

The S7 configuration also enables:

```text
use_source_reliable_boundary_router: true
```

This module moves probability mass from a reliable source class to a nearby target class only when a route is predefined and the physical gate agrees. In the released S7 config, the active route is:

```text
dry_concrete_smooth -> dry_concrete_slight
```

This targets the dry-concrete roughness boundary, where smooth and slight can be visually close.

## 11. Loss

The main supervised signal is 27-class cross entropy:

```text
L = L_CE(y, y_hat) + regularization terms
```

The S7 run additionally used anchor consistency and no-flip constraints to avoid damaging classes that were already strong in the anchor model.

The focus classes were:

```text
water_concrete_slight
wet_concrete_slight
water_concrete_severe
wet_concrete_severe
```

These are difficult because water/wet evidence can hide concrete roughness.

## 12. Output

The model outputs 27 logits and then applies softmax:

```text
p = softmax(logits)
y_hat = argmax_c p_c
```

Metrics are computed on the full 49,500-image RSCD test split.
