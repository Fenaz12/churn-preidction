from langchain_openrouter import ChatOpenRouter

from core.config import settings


def get_llm(temperature: float = 0.2) -> ChatOpenRouter:
    return ChatOpenRouter(
        model=settings.agent_model,
        api_key=settings.openrouter_api_key,
        temperature=temperature,
        max_retries=2,
        openrouter_provider={
            "require_parameters": True,
        },
    )