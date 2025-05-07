from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
import logging
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

router = APIRouter()

UPLOAD_DIR = "./uploaded_manuals"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload_manual")
def upload_manual(file: UploadFile = File(...)):
    try:
        # Remove the upload directory before every upload
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Only accept .txt or .md files
        if not (file.filename.lower().endswith('.txt') or file.filename.lower().endswith('.md')):
            logging.warning(f"Rejected file upload: {file.filename} (not a .txt or .md file)")
            raise HTTPException(status_code=400, detail="Only .txt or .md files are accepted.")

        file_location = os.path.join(UPLOAD_DIR, file.filename)
        logging.info(f"Received file upload: {file.filename}")
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"Saved file to: {file_location}")

        # Use appropriate loader based on file extension
        if file.filename.lower().endswith('.md'):
            loader = UnstructuredMarkdownLoader(file_location)
        else:
            loader = TextLoader(file_location, encoding="utf-8")
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} document(s) from file.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Split document(s) into {len(chunks)} chunk(s).")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logging.info("Initialized HuggingFace embedding model.")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="./chroma_db"
        )
        logging.info("Stored embeddings in Chroma vector database.")
        return JSONResponse(content={"message": "Manual processed and embeddings stored."})
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        logging.error("Exception occurred during manual upload and processing:")
        logging.error(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)
