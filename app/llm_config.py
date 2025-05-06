class LLMConfig:
    def __init__(self, provider: str, model: str, api_key: str = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        if provider == "ollama":
            from langchain_community.llms import Ollama
            self.llm = Ollama(model=model)
        elif provider == "openai":
            from langchain.chat_models import ChatOpenAI
            self.llm = ChatOpenAI(model=model, api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

llm_config = LLMConfig(provider="openai", model="gpt-4.1-nano", api_key="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")
