from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
from langgraph_chatbot import get_response

app = FastAPI()

# Pydantic model for request body
class QuestionRequest(BaseModel):
    question: str

# Function to detect if the question is in Bangla
def is_bangla(text: str) -> bool:
    # Check for Bangla Unicode range (0x0980–0x09FF)
    bangla_pattern = re.compile(r'[\u0980-\u09FF]')
    return bool(bangla_pattern.search(text))

# POST endpoint to handle questions
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question.strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Detect language
    language = "Bangla" if is_bangla(question) else "English"
    
    # Get response from langgraph_chatbot
    try:
        answer = get_response(question, language)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
    
    return {"question": question, "answer": answer, "language": language}

# Run the app with: uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)