import importlib
import logging

_AGENTS = {
    "random": ("agents.meta_agent",        "RandomMetaAgent"),
    "tree":   ("agents.meta_tree_agent",   "DecisionTreeMetaAgent"),
    "forest": ("agents.meta_forest_agent", "RandomForestMetaAgent"),
    "llm":    ("agents.meta_llm_agent",    "LLMMetaAgent"),
}


def make_meta_agent():
    """Создаёт мета-агента согласно META_AGENT в config.py."""
    from config import META_AGENT
    module_name, class_name = _AGENTS.get(META_AGENT, _AGENTS["random"])
    cls = getattr(importlib.import_module(module_name), class_name)
    logging.getLogger("agents").info("Мета-агент: %s (%s)", class_name, META_AGENT)
    return cls()
