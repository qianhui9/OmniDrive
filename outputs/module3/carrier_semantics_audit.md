# Carrier semantics and index audit

Source: current `StreamPETRHead.forward`, `PETRHeadM.forward`, `Petr3D.simple_test_pts`, native `llava_arch.py`; original weights are read-only.

| Combined index | Source | Reliable meaning | Unsupported interpretation |
|---|---|---|---|
| 0–255 | Detection decoder's first `num_extra=256` queries, output projection 256→4096 | Learned detection-branch visual carriers | One carrier = one decoded box or M1 Anchor ID |
| 256 | `can_bus_embed` after detection carrier projection | Ego command/current CAN bus + temporal CAN bus/pose aggregate | Current CAN bus alone |
| 257–512 | Map decoder's first `num_extra=256` queries, output projection 256→4096 | Learned map-branch visual carriers | One carrier = one lane/control point |

The head removes these extra queries before box/map prediction and top-K temporal-memory updates. The proposal confidence and geometry therefore have **no demonstrated index mapping** to these 256 extra queries. Detection Top-K 300 updates temporal proposal memory; it does not order the LLM carriers. Map Top-K similarly updates proposal memory, not carrier selection.

M3 never uses GT, M1 anchor identity, box confidence or box geometry to delete a carrier. The task candidate keeps every object/ego carrier and a leading ordered subset of map carriers for object-relation questions. This is a branch-provenance hypothesis, not a learned or certified relevance mapping. Global, map and unknown questions retain the full carrier. Stale supplied M1 state forces full selection; absent M1 state does not invalidate current directly computed carriers. Auto additionally requires an independent held-out quality certificate.

Norm Top-K and seeded random selection are offline controls, always retaining Ego and sorting retained indices back into original order. Norm is **not** interpreted as detection confidence. No arbitrary prompt reordering, vector synthesis or weight training is used. Token deletion changes subsequent positions and attention and is explicitly approximate.
