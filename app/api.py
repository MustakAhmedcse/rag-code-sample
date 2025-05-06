from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.rag_pipeline import rag_pipeline
from app.utils import is_bangla

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

@router.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    language = "Bangla" if is_bangla(question) else "English"
    try:
        answer = rag_pipeline.get_response(question, language)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
    return {"question": question, "answer": answer, "language": language}
