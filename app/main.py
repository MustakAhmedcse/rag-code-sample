from fastapi import FastAPI
from app.api import router as api_router
from app.manual_upload import router as manual_upload_router

app = FastAPI()

# Include routers
app.include_router(api_router)
app.include_router(manual_upload_router)

# Run the app with: uvicorn app.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
