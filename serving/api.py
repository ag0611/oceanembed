from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from encoder_cnn import CNNEncoder
from encoder_vit import ViTEncoder
from decoder import DualEncoderWithDecoder
from thermocline_head import ThermoclineHead

app = FastAPI(title="OceanEmbed API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

def load_model(checkpoint_path: Optional[str] = None):
    cnn   = CNNEncoder(in_channels=5, embedding_dim=512)
    vit   = ViTEncoder(in_channels=5, embedding_dim=512)
    model = DualEncoderWithDecoder(cnn, vit, num_depths=15)
    thermo = ThermoclineHead(num_depths=15)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        thermo.load_state_dict(ckpt['thermo'])
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint — using random weights (demo mode)")

    model.eval()
    thermo.eval()
    return model, thermo

CKPT = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_model.pt')
model, thermo = load_model(CKPT)

class PredictRequest(BaseModel):
    sst:  List[List[float]]
    sss:  List[List[float]]
    ssh:  List[List[float]]
    u:    List[List[float]]
    v:    List[List[float]]
    lat:  float
    lon:  float
    date: str

class DepthTemp(BaseModel):
    depth_m: int
    temp_c:  float

class PredictResponse(BaseModel):
    lat:               float
    lon:               float
    date:              str
    profile:           List[DepthTemp]
    thermocline_depth: float
    thermocline_grad:  float

@app.get("/health")
def health():
    return {"status": "ok", "model": "OceanEmbed v1.0"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        channels = [req.sst, req.sss, req.ssh, req.u, req.v]
        x = torch.tensor(channels, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            T_pred = model(x)
            td, tg = thermo(T_pred)

        temps = T_pred.squeeze(0).tolist()
        profile = [
            DepthTemp(depth_m=d, temp_c=round(t, 3))
            for d, t in zip(DEPTHS, temps)
        ]

        return PredictResponse(
            lat=req.lat,
            lon=req.lon,
            date=req.date,
            profile=profile,
            thermocline_depth=round(td.item(), 2),
            thermocline_grad=round(tg.item(), 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/depths")
def get_depths():
    return {"depths_m": DEPTHS}