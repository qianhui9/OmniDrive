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

The sole recommended integrated entry and current validation boundary are
documented in [the final route](worldstream_final_route.md).
`configs/worldstream_m1_final.yaml` uses the current ordinary A defaults;
`association_track_prior` remains false. Independent physical verification and
the unified allocator remain intact. DQTrack modes other than off are rejected;
there is no external identity/shadow producer or offline B/C wrapper.
