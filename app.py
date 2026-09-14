from pathlib import Path
import json, subprocess, sys, numpy as np, joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse

ROOT=Path(__file__).parent
MODEL=ROOT/"models/landslide_model.pkl"; FEAT=ROOT/"models/features.json"
if not MODEL.exists() or not FEAT.exists(): subprocess.run([sys.executable,str(ROOT/"train.py")],check=True)
model=joblib.load(MODEL); features=json.loads(FEAT.read_text())
app=FastAPI(title="TERRA SAFE AI")

@app.get("/")
def home(): return FileResponse(ROOT/"static/index.html")

@app.get("/health")
def health(): return {"project":"TERRA SAFE AI","status":"running","model":"Random Forest"}

@app.get("/metrics")
def metrics(): return json.loads((ROOT/"outputs/metrics.json").read_text())

@app.get("/api/zones")
def zones():
    return {"zones":[
      {"name":"North Ridge","lat":11.030,"lng":76.950,"risk":"HIGH","score":88},
      {"name":"Hill View","lat":11.010,"lng":76.975,"risk":"HIGH","score":76},
      {"name":"East Valley","lat":10.990,"lng":76.940,"risk":"MODERATE","score":54},
      {"name":"South Road","lat":10.970,"lng":76.965,"risk":"LOW","score":22}],
      "roads":[
      {"name":"Mountain Road","risk":"HIGH","status":"Avoid"},
      {"name":"Valley Link Road","risk":"MODERATE","status":"Use caution"},
      {"name":"City Bypass","risk":"LOW","status":"Safer"}]}

@app.post("/predict")
def predict(data:dict):
    vals=[float(data[f]) for f in features]
    p=float(model.predict_proba(np.array(vals).reshape(1,-1))[0][1])
    risk="LOW" if p<=.30 else "MODERATE" if p<=.60 else "HIGH"
    return {"probability":round(p,4),"risk_score":round(p*100,1),"risk":risk,
            "message":"High landslide risk. Follow local emergency guidance." if risk=="HIGH" else
                      "Monitor rainfall, slope and terrain conditions." if risk=="MODERATE" else
                      "Conditions look relatively stable."}
