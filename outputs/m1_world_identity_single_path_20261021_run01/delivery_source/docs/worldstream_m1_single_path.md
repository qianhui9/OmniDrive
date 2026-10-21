# M1 single identity allocation

`WorldStateManager._commit()` reads source-bound active/dormant inputs through
`association_tracks.candidates()`. Each object is predicted once by
`association_tracks.predict()`. The resulting centres/budgets enter
`association.associate()` directly. It builds one legal distance matrix, rejects
unresolved endpoint competition, and calls `unmatched_assignment()` once.
The returned pairs are frozen as a tuple before observation/physical evidence
processing. No later producer can replace, add or veto an identity pair.

Prediction uses continuous bounded history (median slopes and residual/scatter),
otherwise the latest observed centre or an independently qualified existing
physical motion prediction. Unqualified prediction budgets include declared
position uncertainty plus `velocity_uncertainty_mps * observation_age`. A fit
adds measured residual/scatter. Exceeding the fixed geometric budget rejects the
candidate. Neither the fit nor an ordinary identity is a physical certificate.

Dormant records contain geometry/source metadata only. They pass the same
prediction/gates/allocation; there is no recovery algorithm. A reappearing ID
receives new actual state without old verification times, features or certificates.
History publication and pruning remain inside the existing atomic transaction.

The primary config is `configs/worldstream_m1_world_identity.yaml`. The
`association_track_prior` switch controls bounded history, not another allocator.
With it disabled, the same allocator uses current stored anchors and the same
predictor's conservative short-history rule.

Retired assignment options fail explicitly when constructing a manager:
`association_observation_guard=True`, `search_dynamic_motion=True`, identity
tracking policies other than `shadow`, DQTrack `hint`/`calibrated_tiebreak`, and
DQTrack identity-policy qualification files. External `shadow` observations and
independent physical measurement/issuer/validator functions remain available.
Old hint/search YAMLs have been migrated; removed detailed-diagnostic config
fields are no longer accepted. There is no fallback to an old assignment path.

Validation and version boundary: see
`outputs/m1_world_identity_single_path_20261021_run01/experiment_report.md`.
Only one real cached replay was run. It exposed unsafe dormant matches; the
final common time-error budget was corrected and unit-tested afterwards.
Final real-data quality is **not validated** and deployment acceptance is No-Go.
