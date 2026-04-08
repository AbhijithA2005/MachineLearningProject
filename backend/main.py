from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from logic import get_ml_info

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProblemRequest(BaseModel):
    description: str

@app.post("/predict")
async def predict_paradigm(request: ProblemRequest):
    if not request.description:
        raise HTTPException(status_code=400, detail="Description is required")
    
    if get_ml_info.__code__.co_argcount == 1:
        result = get_ml_info(request.description)
    else:
        # Fallback if I change signature
        result = get_ml_info(request.description)
    return result

@app.get("/suggestions")
async def get_suggestions():
    return [
        "Predict customer churn based on transaction history",
        "Group products by sales patterns and seasonal trends",
        "Train an agent to optimize logistics and routing",
        "Forecast daily stock prices using sentiment analysis",
        "Identify anomalous network traffic without labels",
        "Optimize energy consumption in a smart grid system"
    ]

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
