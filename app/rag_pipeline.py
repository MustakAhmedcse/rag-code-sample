from app.llm_config import llm_config
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import Dict, Any
from app.utils import is_relevant_to_retailer_app

class RAGPipeline:
    def __init__(self, llm_config):
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        prompt_template = """
        You are a helpful assistant for the Banglalink Retailer App. Answer the user's question only if it is directly related to the Banglalink Retailer App's features, functionality, or usage. Provide clear, step-by-step guidance in {language}. If the question is unrelated to the Banglalink Retailer App (e.g., general questions like \"What is AI?\"), respond with: \"I can only assist with questions related to the Banglalink Retailer App. Please ask a relevant question.\" If the question is unclear or no relevant context is found, ask for clarification.

        Context: {context}

        Question: {question}

        Answer:
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "language"]
        )
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
        class ChatbotState(Dict[str, Any]):
            question: str
            language: str
            answer: str
            needs_clarification: bool
        def chatbot_node(state: ChatbotState) -> ChatbotState:
            if not is_relevant_to_retailer_app(state["question"]):
                state["answer"] = "I can only assist with questions related to the Banglalink Retailer App. Please ask a relevant question."
                state["needs_clarification"] = True
                return state
            context = self.retriever.invoke(state["question"])
            if not context:
                state["answer"] = f"No relevant information found in {state['language']}. Please clarify your question."
                state["needs_clarification"] = True
                return state
            answer = self.rag_chain.invoke({"question": state["question"], "language": state["language"]})
            state["answer"] = answer.strip()
            state["needs_clarification"] = False
            return state
        workflow = StateGraph(ChatbotState)
        workflow.add_node("chatbot", chatbot_node)
        workflow.set_entry_point("chatbot")
        workflow.add_edge("chatbot", END)
        self.app = workflow.compile()
    def get_response(self, question: str, language: str) -> str:
        result = self.app.invoke({"question": question, "language": language, "answer": "", "needs_clarification": False})
        return result["answer"]

rag_pipeline = RAGPipeline(llm_config)
