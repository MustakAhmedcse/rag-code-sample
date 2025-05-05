from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load the vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# Initialize Ollama LLM
llm = OllamaLLM(model="llama3")

# Define a prompt template
prompt_template = """
You are a helpful assistant for the Banglalink Retailer App. Answer the user's question based on the provided context from the user manual. Provide clear, step-by-step guidance. If the question is unclear, ask for clarification.

Context: {context}

Question: {question}

Answer:
"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Create the RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": prompt}
)

# Function to get response
def get_response(question, language="English"):
    result = rag_chain.invoke({"query": question})
    answer = result["result"]
    # If language is Bangla, add a note to respond in Bangla
    if language == "Bangla":
        return f"উত্তর (বাংলায়):\n{answer}"
    return answer

# Test the pipeline
if __name__ == "__main__":
    question = "How do I register a device in the Banglalink Retailer App?"
    response = get_response(question, language="English")
    print("English Response:", response)
    
    # Test in Bangla
    question_bn = "বাংলালিংক রিটেইলার অ্যাপে ডিভাইস কীভাবে নিবন্ধন করব?"
    response_bn = get_response(question_bn, language="Bangla")
    print("Bangla Response:", response_bn)