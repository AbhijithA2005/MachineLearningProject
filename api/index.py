from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# Use relative import for logic if running in api/ folder
try:
    from .logic import get_ml_info
except ImportError:
    from logic import get_ml_info

app = FastAPI()

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProblemRequest(BaseModel):
    description: str

@app.post("/api/predict")
async def predict_paradigm(request: ProblemRequest):
    if not request.description:
        raise HTTPException(status_code=400, detail="Description is required")
    try:
        result = get_ml_info(request.description)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/suggestions")
async def get_suggestions():
    return [
        "Predict customer churn based on transaction history",
        "Group products by sales patterns and seasonal trends",
        "Train an agent to optimize logistics and routing",
        "Forecast daily stock prices using sentiment analysis",
        "Identify anomalous network traffic without labels",
        "Optimize energy consumption in a smart grid system"
    ]

@app.get("/api/health")
async def health():
    return {"status": "healthy", "runtime": "vercel"}

# Vercel needs the 'app' object to be available at the module level
