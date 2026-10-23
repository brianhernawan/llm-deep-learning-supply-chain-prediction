"""FastAPI service. Run from the project root:  uvicorn api:app --host 0.0.0.0 --port 8010
Try it in the browser: http://localhost:8010/docs
"""
import os
from datetime import date
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import predict

load_dotenv()
app = FastAPI(title="Carrier Advisor API")


class QuoteRequest(BaseModel):
    origin_warehouse: str = Field(examples=["Warehouse_LA"])
    destination: str = Field(examples=["Boston"])
    weight_kg: float = Field(gt=0, examples=[25])
    distance_miles: Optional[float] = Field(default=None, gt=0)
    ship_date: Optional[date] = None
    model: Literal["ensemble", "xgboost", "keras"] = "ensemble"


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def llm_ready():
    key = os.getenv("GROQ_API_KEY", "")
    return bool(key) and not key.startswith("your-")


@app.get("/health")
def health():
    predict.load_models()
    return {"status": "ok", "llm_ready": llm_ready()}


@app.post("/quote")
def quote(req: QuoteRequest):
    try:
        table = predict.quote(req.origin_warehouse, req.destination, req.weight_kg,
                              req.distance_miles, req.ship_date, req.model)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))
    return {"quotes": table.to_dict(orient="records"), "cheapest": table["Carrier"].iloc[0]}


@app.post("/ask")
def ask(q: Question):
    if not llm_ready():
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not set in .env")
    from agent import ask as agent_ask  # load LangChain only when needed
    try:
        return agent_ask(q.question)
    except Exception as err:  # rate limit or network problem: tell the user, don't crash
        raise HTTPException(status_code=502, detail=f"LLM call failed: {err}")
