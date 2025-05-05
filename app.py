import logging
import re
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Pydantic model for request body
class QuestionRequest(BaseModel):
    question: str

# LLM Configuration
class LLMConfig:
    def __init__(self, provider: str = "ollama", model: str = "llama3", api_key: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        logger.info(f"Initializing LLM with provider: {self.provider}, model: {self.model}")
        if self.provider == "ollama":
            return OllamaLLM(model=self.model)
        elif self.provider == "openai":
            if not self.api_key:
                logger.error("Open AI API key not provided")
                raise ValueError("Open AI API key is required")
            return OpenAI(model=self.model, api_key=self.api_key)
        else:
            logger.error(f"Unsupported LLM provider: {self.provider}")
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

# Function to check if the question is relevant to Banglalink Retailer App
def is_relevant_to_retailer_app(question: str) -> bool:
    logger.info(f"Checking relevance of question: {question}")
    # Keywords related to Banglalink Retailer App
    retailer_keywords = [
        "retailer", "banglalink", "app", "profile", "device", "registration",
        "feedback", "voice of retailer", "commission", "lifting", "dashboard",
        "login", "logout", "edit", "submit", "form", "operator", "category"
    ]
    question_lower = question.lower()
    is_relevant = any(keyword in question_lower for keyword in retailer_keywords)
    logger.info(f"Question relevance result: {is_relevant}")
    return is_relevant

# RAG Pipeline Setup
class RAGPipeline:
    def __init__(self, llm_config: LLMConfig):
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.app = None
        self._initialize_pipeline(llm_config)

    def _initialize_pipeline(self, llm_config: LLMConfig):
        logger.info("Initializing RAG Pipeline...")
        
        # Load vector store
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Loading vector store from ./chroma_db")
        self.vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        logger.info("Vector store and retriever initialized")

        # Define prompt template with strict instructions
        prompt_template = """
        You are a helpful assistant for the Banglalink Retailer App. Answer the user's question only if it is directly related to the Banglalink Retailer App's features, functionality, or usage. Provide clear, step-by-step guidance in {language}. If the question is unrelated to the Banglalink Retailer App (e.g., general questions like "What is AI?"), respond with: "I can only assist with questions related to the Banglalink Retailer App. Please ask a relevant question." If the question is unclear or no relevant context is found, ask for clarification.

        Context: {context}

        Question: {question}

        Answer:
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "language"]
        )
        logger.info("Prompt template initialized")

        # Create RAG chain
        self.rag_chain = (
            {
                "context": lambda x: "\n".join(doc.page_content for doc in self.retriever.invoke(x["question"])),
                "question": lambda x: x["question"],
                "language": lambda x: x["language"]
            }
            | prompt
            | llm_config.llm
            | StrOutputParser()
        )
        logger.info("RAG chain initialized")

        # Define LangGraph state
        class ChatbotState(Dict[str, Any]):
            question: str
            language: str
            answer: str
            needs_clarification: bool

        # Define chatbot node
        def chatbot_node(state: ChatbotState) -> ChatbotState:
            logger.info(f"Processing question: {state['question']} in language: {state['language']}")
            
            # Check if the question is relevant
            if not is_relevant_to_retailer_app(state["question"]):
                logger.warning("Question is not relevant to Banglalink Retailer App")
                state["answer"] = "I can only assist with questions related to the Banglalink Retailer App. Please ask a relevant question."
                state["needs_clarification"] = True
                return state

            context = self.retriever.invoke(state["question"])
            if not context:
                logger.warning("No relevant context found")
                state["answer"] = f"No relevant information found in {state['language']}. Please clarify your question."
                state["needs_clarification"] = True
                return state
            
            logger.info("Context retrieved, generating answer...")
            answer = self.rag_chain.invoke({"question": state["question"], "language": state["language"]})
            # Ensure the answer is not repeated
            state["answer"] = answer.strip()
            state["needs_clarification"] = False
            logger.info("Answer generated successfully")
            return state

        # Build the LangGraph workflow
        workflow = StateGraph(ChatbotState)
        workflow.add_node("chatbot", chatbot_node)
        workflow.set_entry_point("chatbot")
        workflow.add_edge("chatbot", END)
        self.app = workflow.compile()
        logger.info("LangGraph workflow compiled")

    def get_response(self, question: str, language: str) -> str:
        logger.info(f"Invoking LangGraph app for question: {question}")
        result = self.app.invoke({"question": question, "language": language, "answer": "", "needs_clarification": False})
        logger.info(f"Response received: \n{result['answer']}")
        return result["answer"]

# Function to detect if the question is in Bangla
def is_bangla(text: str) -> bool:
    logger.info(f"Detecting language for text: {text}")
    bangla_pattern = re.compile(r'[\u0980-\u09FF]')
    result = bool(bangla_pattern.search(text))
    logger.info(f"Language detection result: {'Bangla' if result else 'English'}")
    return result

# Initialize LLM and RAG pipeline
llm_config = LLMConfig(provider="ollama", model="llama3")  # Change to "openai" and provide api_key if needed
rag_pipeline = RAGPipeline(llm_config)

# POST endpoint to handle questions
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question.strip()
    logger.info(f"Received question: {question}")
    
    if not question:
        logger.error("Question is empty")
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Detect language
    language = "Bangla" if is_bangla(question) else "English"
    
    # Get response from RAG pipeline
    try:
        answer = rag_pipeline.get_response(question, language)
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
    
    logger.info(f"Returning response for question: {question}")
    return {"question": question, "answer": answer, "language": language}

# Run the app with: uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)