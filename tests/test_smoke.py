from autointerp_agent.config import load_config
from autointerp_agent.skills import SkillRegistry
from autointerp_agent.tools import ToolRouter


def test_skill_registry_loads():
    registry = SkillRegistry.from_dir("skills")
    assert "logit-lens" in registry.skills
    assert "natural-language-autoencoders" in registry.skills
    assert "predictive-concept-decoders" in registry.skills
    assert len(registry.skills) >= 10


def test_config_loads_default_file():
    config = load_config("configs/agent.yaml")
    assert config.model_name
    assert "black-box-auditing" in config.default_skills


def test_tool_router_exposes_core_tools():
    registry = SkillRegistry.from_dir("skills")
    router = ToolRouter(registry)
    names = {tool["function"]["name"] for tool in router.get_tool_specs_for_llm()}
    assert {"plan", "list_skills", "read_skill", "bash", "read_file"}.issubset(names)
