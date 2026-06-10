from .config import AgentRuntimeConfig


def build_agent():
    config = AgentRuntimeConfig.from_env()
    try:
        from .llm_agent import OpenAILangGraphITSupportAgent
    except Exception as exc:
        raise ImportError(
            "LLM agent dependencies are missing. Install `openai` and `langgraph` in the active environment."
        ) from exc

    return OpenAILangGraphITSupportAgent(config=config)