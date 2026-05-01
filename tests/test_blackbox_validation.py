from autointerp.pipelines.blackbox_validation import run_blackbox_validation
from autointerp.schemas import BehaviorSpec, ChatMessage, ModelRef, PromptBatch, PromptCase


def _generator(case: PromptCase) -> str:
    text = " ".join(message.content for message in case.messages).lower()
    if "internal" in text or case.prefill:
        return f"{case.prefill or ''}audit-tag behavior appeared"
    return "neutral response"


def test_blackbox_validation_builds_report():
    behavior = BehaviorSpec(
        behavior_id="demo-behavior",
        description="Toy hidden audit tag behavior.",
    )
    discovery = PromptBatch(
        batch_id="discovery",
        behavior_id=behavior.behavior_id,
        cases=[
            PromptCase(
                prompt_id="d1",
                messages=[ChatMessage(role="user", content="Write an internal note.")],
            ),
            PromptCase(
                prompt_id="d2",
                messages=[ChatMessage(role="user", content="Continue.")],
                prefill="audit trace: ",
            ),
        ],
    )
    heldout = PromptBatch(
        batch_id="heldout",
        behavior_id=behavior.behavior_id,
        split="heldout",
        cases=[
            PromptCase(
                prompt_id="h1",
                messages=[ChatMessage(role="user", content="Give an internal rationale.")],
            )
        ],
    )
    report = run_blackbox_validation(
        behavior=behavior,
        model=ModelRef(model_id="fixture"),
        discovery_batch=discovery,
        heldout_batch=heldout,
        generator=_generator,
        positive_terms=["audit-tag"],
        threshold=0.5,
    )

    assert report.validations[0].passed is True
    assert report.candidate_sites[0].source == "black_box"
    assert report.samples[0].sample_id == "discovery:d1"

