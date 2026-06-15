from langchain_openai import ChatOpenAI
from core.config import settings

def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """
    Returns a configured LangChain Chat model connected via OpenRouter.
    """
    return ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        model=settings.agent_model,
        temperature=temperature,
        default_headers={
            "HTTP-Referer": "https://your-churn-app.internal", 
            "X-Title": "Churn Agent Orchestrator"
        }
    )