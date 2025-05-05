from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import Dict, Any
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize vector store and LLM
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
llm = OllamaLLM(model="llama3")

# Define a prompt template
prompt_template = """
You are a helpful assistant for the Banglalink Retailer App. Answer the user's question based on the provided context from the user manual. Provide clear, step-by-step guidance in {language}. If the question is unclear or no relevant context is found, ask for clarification and provide a general response.

Context: {context}

Question: {question}

Answer:
"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question", "language"]
)

# Create the RAG chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
rag_chain = (
    {
        "context": lambda x: "\n".join(doc.page_content for doc in retriever.invoke(x)),
        "question": lambda x: x,
        "language": lambda x: x
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Define the state for LangGraph
class ChatbotState(Dict[str, Any]):
    question: str
    language: str
    answer: str
    needs_clarification: bool

# Define the chatbot node
def chatbot_node(state: ChatbotState) -> ChatbotState:
    question = state["question"]
    language = state["language"]
    context = retriever.invoke(question)
    if not context:
        state["answer"] = f"No relevant information found in {language}. Please clarify your question."
        state["needs_clarification"] = True
        return state
    answer = rag_chain.invoke(question)
    state["answer"] = answer
    state["needs_clarification"] = False
    return state

# Build the graph
workflow = StateGraph(ChatbotState)
workflow.add_node("chatbot", chatbot_node)
workflow.set_entry_point("chatbot")
workflow.add_edge("chatbot", END)
app = workflow.compile()

# Function to get response
def get_response(question: str, language: str = "English") -> str:
    result = app.invoke({"question": question, "language": language, "answer": "", "needs_clarification": False})
    return result["answer"]

# Test the pipeline
if __name__ == "__main__":
    question = "How do I register a device in the Banglalink Retailer App?"
    response = get_response(question, language="English")
    print("English Response:", response)
    
    # Test in Bangla
    question_bn = "বাংলালিংক রিটেইলার অ্যাপে ডিভাইস কীভাবে নিবন্ধন করব?"
    response_bn = get_response(question_bn, language="Bangla")
    print("Bangla Response:", response_bn)