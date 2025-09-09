from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="RejuvBridge Inference API")


class PredictRequest(BaseModel):
    # In practice: base64 image, tile references, or precomputed embeddings
    payload: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    # Stub: echo
    return {"ok": True, "received": len(req.payload)}


def run_api(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)

