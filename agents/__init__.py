import importlib
import logging

_AGENTS = {
    "llm":  ("agents.meta_llm_agent",  "LLMMetaAgent"),
    "rule": ("agents.meta_rule_agent", "RuleMetaAgent"),
}


def make_meta_agent():
    """Создаёт мета-агента согласно META_AGENT в config.py."""
    from config import META_AGENT
    module_name, class_name = _AGENTS.get(META_AGENT, _AGENTS["rule"])
    cls = getattr(importlib.import_module(module_name), class_name)
    logging.getLogger("agents").info("Мета-агент: %s (%s)", class_name, META_AGENT)
    return cls()
