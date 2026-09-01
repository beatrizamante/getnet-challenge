from pydantic import BaseModel, Field


class PromptCatalog(BaseModel):
    router_system: str = Field(min_length=1)
    knowledge_system: str = Field(min_length=1)
    customer_support_system: str = Field(min_length=1)
    input_guardrail_classifier: str = Field(min_length=1)
