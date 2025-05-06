class LLMConfig:
    def __init__(self, provider: str, model: str, api_key: str = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        if provider == "ollama":
            from langchain_community.llms import Ollama
            self.llm = Ollama(model=model)
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model=model, api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

llm_config = LLMConfig(provider="ollama", model="llama3")
