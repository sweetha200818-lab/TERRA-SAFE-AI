from pathlib import Path
import json, numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix

ROOT=Path(__file__).parent
(ROOT/"models").mkdir(exist_ok=True); (ROOT/"outputs").mkdir(exist_ok=True)
features=["rainfall","soil_moisture","temperature","humidity","slope","elevation","land_cover","distance_to_road","historical_landslides","terrain_roughness"]
rng=np.random.default_rng(42); n=1800
rain=rng.uniform(0,300,n); soil=rng.uniform(.05,.95,n); temp=rng.uniform(12,40,n)
hum=rng.uniform(25,100,n); slope=rng.uniform(0,60,n); elev=rng.uniform(50,2800,n)
cover=rng.integers(1,6,n); dist=rng.uniform(0,5000,n); hist=rng.integers(0,4,n); rough=rng.uniform(0,1,n)
score=.010*rain+2.6*soil+.055*slope+.00025*elev+.65*hist+1.1*rough+.20*hum-.00008*dist-.018*temp+.10*(cover==1)+rng.normal(0,.9,n)
y=(score>np.percentile(score,58)).astype(int)
X=np.column_stack([rain,soil,temp,hum,slope,elev,cover,dist,hist,rough])
Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=250,random_state=42,class_weight="balanced").fit(Xt,yt)
pred=model.predict(Xv); prob=model.predict_proba(Xv)[:,1]
m={"accuracy":round(float(accuracy_score(yv,pred)),4),"precision":round(float(precision_score(yv,pred,zero_division=0)),4),"recall":round(float(recall_score(yv,pred,zero_division=0)),4),"f1":round(float(f1_score(yv,pred,zero_division=0)),4),"roc_auc":round(float(roc_auc_score(yv,prob)),4),"confusion_matrix":confusion_matrix(yv,pred).tolist()}
joblib.dump(model,ROOT/"models/landslide_model.pkl")
(ROOT/"models/features.json").write_text(json.dumps(features,indent=2))
(ROOT/"outputs/metrics.json").write_text(json.dumps(m,indent=2))
print("Model trained successfully."); print(json.dumps(m,indent=2))
