from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any
import json

from pipeline import run_analysis
from config import settings

app = FastAPI(
    title="NutriManas API",
    description="AI-powered food quality & nutrition analyzer",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserProfile(BaseModel):
    age: int = 30
    gender: str = "male"
    weight_kg: float = 70.0
    height_cm: float = 170.0
    bmi: float | None = None
    activity_level: str = "moderate"
    health_goal: str = "general wellness"
    health_conditions: list[str] = []


class AnalyzeResponse(BaseModel):
    food_name: str
    classification: dict[str, Any]
    nutrition: dict[str, Any]
    quality: dict[str, Any]
    health: dict[str, Any]
    explanation: dict[str, Any]
    error: str | None


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "NutriManas API"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_food(
    file: UploadFile = File(...),
    profile: str | None = None,
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, png, webp)")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    user_profile = None
    if profile:
        try:
            profile_data = json.loads(profile)
            p = UserProfile(**profile_data)
            if p.bmi is None and p.weight_kg and p.height_cm:
                p.bmi = round(p.weight_kg / ((p.height_cm / 100) ** 2), 1)
            user_profile = p.model_dump()
        except Exception:
            user_profile = None

    result = await run_analysis(image_bytes, user_profile)
    return JSONResponse(content=result)


@app.get("/models/status")
async def models_status():
    import ollama as ol
    try:
        models = ol.list()
        available = [m["name"] for m in models.get("models", [])]
    except Exception:
        available = []

    return {
        "ollama_models": available,
        "vision_model": settings.ollama_vision_model,
        "text_model": settings.ollama_text_model,
        "vision_ready": any(settings.ollama_vision_model in m for m in available),
        "text_ready": any(settings.ollama_text_model in m for m in available),
    }
