# priors/legacy

Retired phenomenon priors. A "prior" pre-filled parts of an investigation plan
for a *named* phenomenon (e.g. `ioi.yaml` seeded the IOI circuit task), loaded
by the `retrieve_prior` Stage-0 tool from `priors/<id>.yaml`.

These are kept here for reference only and are **not active**: the loader globs
`priors/*.yaml` non-recursively, so nothing in this `legacy/` subdir is
discoverable, and `retrieve_prior` is gated behind
`create_stage0_tools(include_priors=False)` — off by default and not enabled by
any caller.

**Why retired.** Per-phenomenon priors give specific questions special built-in
support, which is exactly what the planner is meant *not* to have — its system
prompt states "You have NO phenomenon priors available — reason from first
principles." Keeping a live `ioi.yaml` contradicted that and made IOI an unfair
benchmark for the scaffold's generality. The agent now reasons every plan from
scratch, the same for every question.

To revive one for experimentation, move it back to `priors/` and call
`create_stage0_tools(include_priors=True)`.
