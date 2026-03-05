import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Optional

# Import modules (we will refactor these next to return raw data)
import sys
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from modules.classification import run_classification
    from modules.correlation import compute_correlation
    from modules.regression import run_regression
    from modules.gradient_descent import run_gd
    from modules.model_selection import run_model_selection
except ImportError:
    pass # Will fix imports as we refactor

app = FastAPI(title="Smart ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Smart ML API is running"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Read to get metadata
    try:
        df = pd.read_csv(file_path)
        columns = df.columns.tolist()
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        return {
            "filename": file.filename,
            "rows": len(df),
            "columns": columns,
            "dtypes": dtypes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ProblemRequest(BaseModel):
    filename: str
    target_col: Optional[str] = None
    target_threshold: Optional[float] = None
    feature_cols: List[str] = []
    cat_cols: List[str] = []
    num_cols: List[str] = []
    # Problem 4 specific
    iterations: Optional[int] = 1000
    learning_rates: Optional[List[float]] = [0.1, 0.01, 0.001]
    # Problem 5 specific
    reduced_features: Optional[List[str]] = []
    interact_a: Optional[str] = None
    interact_b: Optional[str] = None

def get_df(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found. Please upload again.")
    return pd.read_csv(path)

def clean_float(val):
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    elif isinstance(val, (int, np.integer)):
        return int(val)
    return val

def clean_data(obj):
    if isinstance(obj, dict):
        return {str(k): clean_data(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        return [clean_data(x) for x in obj]
    elif pd.isna(obj):  # Catch pandas NA/NaNs
        return None
    else:
        return clean_float(obj)

def prepare_df_and_req(df: pd.DataFrame, req: ProblemRequest) -> pd.DataFrame:
    effective_features = req.feature_cols if req.feature_cols else req.num_cols
    if not effective_features:
        effective_features = []
        
    valid_cats = []
    valid_nums = []
    for c in effective_features:
        if c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                if df[c].nunique() < 15:
                    valid_cats.append(c)
                else:
                    valid_nums.append(c)
            else:
                if df[c].nunique() < 15:
                    valid_cats.append(c)
                
    if req.cat_cols:
        req.cat_cols = [c for c in req.cat_cols if c in valid_cats]
    else:
        req.cat_cols = valid_cats
        
    if req.num_cols:
        req.num_cols = [c for c in req.num_cols if c in valid_nums]
    else:
        req.num_cols = valid_nums
        
    req.feature_cols = req.cat_cols + req.num_cols
    
    cols_to_keep = req.feature_cols.copy() if req.feature_cols else req.num_cols.copy()
    if req.target_col and req.target_col in df.columns and req.target_col not in cols_to_keep:
        cols_to_keep.append(req.target_col)
        
    return df[cols_to_keep]

@app.post("/api/problem1")
def problem1(req: ProblemRequest):
    df = get_df(req.filename)
    df = prepare_df_and_req(df, req)
    if not req.target_col or (not req.feature_cols and not req.num_cols):
        raise HTTPException(status_code=400, detail="Target and feature columns required")
    
    if req.target_threshold is not None:
        if pd.api.types.is_numeric_dtype(df[req.target_col]):
            df[req.target_col] = (df[req.target_col] >= req.target_threshold).astype(int)
    
    # We will adjust the module to just return the 5 items
    try:
        class_dist, base_metrics, base_cm, res_metrics, res_cm = run_classification(
            df, req.target_col, req.feature_cols, req.cat_cols
        )
        return {
            "class_dist": class_dist.to_dict(),
            "base_metrics": base_metrics,
            "base_cm": base_cm.tolist(),
            "res_metrics": res_metrics,
            "res_cm": res_cm.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/problem2")
def problem2(req: ProblemRequest):
    df = get_df(req.filename)
    df = prepare_df_and_req(df, req)
    if not req.num_cols:
        raise HTTPException(status_code=400, detail="Numerical columns required")
    
    try:
        corr, top_pos, top_neg = compute_correlation(df, req.num_cols)
        return {
            "z": corr.values.tolist(),
            "labels": corr.columns.tolist(),
            "top_pos": top_pos.to_dict(orient="records"),
            "top_neg": top_neg.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/problem3")
def problem3(req: ProblemRequest):
    # This module returns a statsmodel object, arrays, etc. We will need to serialize them.
    df = get_df(req.filename)
    df = prepare_df_and_req(df, req)
    try:
        model, y, y_pred, residuals, mse, rmse, vif_data = run_regression(
            df, req.target_col, req.feature_cols, req.cat_cols, req.num_cols
        )
        
        # We need to send down samples for the plots since full arrays (17k) are big
        # Let's take a random sample of 1000 for scatter plots
        sample_size = min(1000, len(y))
        indices = np.random.choice(len(y), sample_size, replace=False)
        
        return clean_data({
            "metrics": {
                "rsquared": model.rsquared,
                "rsquared_adj": model.rsquared_adj,
                "mse": mse,
                "rmse": rmse
            },
            "vif": vif_data.to_dict(orient="records"),
            "plots": {
                "y_sampled": y.iloc[indices].tolist(),
                "y_pred_sampled": y_pred.iloc[indices].tolist(),
                "residuals_sampled": residuals.iloc[indices].tolist()
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/problem4")
def problem4(req: ProblemRequest):
    df = get_df(req.filename)
    df = prepare_df_and_req(df, req)
    try:
        results = run_gd(
            df, req.target_col, req.feature_cols, req.cat_cols, req.num_cols,
            req.learning_rates, req.iterations
        )
        
        serializable_results = {}
        for alpha, res in results.items():
            serializable_results[str(alpha)] = {
                "cost_history": res["cost_history"].tolist(),
                "final_cost": res["final_cost"]
            }
        return clean_data(serializable_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/problem5")
def problem5(req: ProblemRequest):
    df = get_df(req.filename)
    df = prepare_df_and_req(df, req)
    try:
        comparison_df, models = run_model_selection(
            df, req.target_col, req.feature_cols, req.cat_cols, req.num_cols,
            req.reduced_features, req.interact_a, req.interact_b
        )
        return clean_data({
            "comparison": comparison_df.to_dict(orient="records")
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
