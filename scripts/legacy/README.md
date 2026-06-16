# scripts/legacy

One-off scripts from early, **manual** investigation runs (IOI and
agentic-misalignment). Kept for reference only — nothing in the package, docs,
CI, or other scripts imports or invokes them.

They sit *outside* the investigation pipeline: they do raw forward passes and
write to `scratch/` by hand, rather than going through the gated tools
(`compute_and_commit_metric`, `record_capture`) and the provenance / grounding
layer. The real Stage-0 foundations live in `src/autointerp/pipelines/` and the
phenomenon priors in `priors/`.

These were committed incidentally (swept into an unrelated feature commit) and
moved here to keep the top-level `scripts/` to genuine repo tooling
(`validate_skills.py`, `install_torch.py`). To reproduce a real investigation,
use `autointerp` / the pipeline — not these.
