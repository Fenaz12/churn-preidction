from pydantic import BaseModel, Field
from typing import Optional, TypedDict
from langchain_openrouter import ChatOpenRouter
from core.config import settings


class CustomerRisk(BaseModel):
    city: str = Field(..., description="The city where the customer resides.")



llm = ChatOpenRouter(
    model="gpt-4o-mini",
    api_key=settings.openrouter_api_key
)

structured_llm = llm.with_structured_output(CustomerRisk)

response = structured_llm.invoke("What is the capital of France?")

print(response)