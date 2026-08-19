from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from haversine import haversine, Unit
import joblib

# FastAPI Instance Init karein
app = FastAPI(
    title="Causal Inference DML API",
    description="Double Machine Learning API for predicting treatment effects",
    version="1.0"
)

# ----------------------------------------------------
# 1. Models & Preprocessors Load Karein
# ----------------------------------------------------
try:
    col_trans = joblib.load('column_transformer.pkl')
    category_map = joblib.load('category_mapping.pkl')

    dml_models = {
        'price': joblib.load('dml_model_price.pkl'),
        'freight_value': joblib.load('dml_model_freight_value.pkl'),
        'payment_value': joblib.load('dml_model_payment_value.pkl')
    }
    print("Saare models aur preprocessors successfully load ho gaye hain!")
except Exception as e:
    print(f"Error loading model files: {e}")

# ----------------------------------------------------
# 2. Input Data Schema (Pydantic Model)
# ----------------------------------------------------
class OrderInput(BaseModel):
    product_category_name_english: str
    product_description_lenght: float
    product_photos_qty: float
    product_weight_g: float
    product_length_cm: float
    product_height_cm: float
    product_width_cm: float
    purchase_month: int
    purchase_day_of_week: int
    purchase_hour: int
    customer_city: str
    seller_city: str
    customer_state: str
    seller_state: str
    customer_lat: float
    customer_lng: float
    seller_lat: float
    seller_lng: float

# ----------------------------------------------------
# 3. Preprocessing Function
# ----------------------------------------------------
def preprocess_input(data: OrderInput):
    input_dict = data.dict()
    df_single = pd.DataFrame([input_dict])

    # Category mapping
    raw_cat = df_single['product_category_name_english'].iloc[0]
    df_single['product_category_name'] = category_map.get(raw_cat, 'Other')

    # Feature Engineering
    df_single['product_volume_cm3'] = (
        df_single['product_description_lenght'] * 
        df_single['product_length_cm'] * 
        df_single['product_height_cm']
    )
    df_single['is_same_city'] = int(
        df_single['customer_city'].str.lower().iloc[0] == df_single['seller_city'].str.lower().iloc[0]
    )
    df_single['is_same_state'] = int(
        df_single['customer_state'].str.lower().iloc[0] == df_single['seller_state'].str.lower().iloc[0]
    )

    # Distance calculation (Treatment Variable T)
    cust_loc = (df_single['customer_lat'].iloc[0], df_single['customer_lng'].iloc[0])
    sell_loc = (df_single['seller_lat'].iloc[0], df_single['seller_lng'].iloc[0])
    distance_km = haversine(cust_loc, sell_loc, unit=Unit.KILOMETERS)

    # Confounder Features (X)
    confounder_cols = [
        'product_category_name', 'product_description_lenght', 'product_photos_qty',
        'product_weight_g', 'purchase_month', 'purchase_day_of_week',
        'purchase_hour', 'product_volume_cm3', 'is_same_city', 'is_same_state'
    ]
    X_raw = df_single[confounder_cols]
    
    # Transformation
    X_transformed = col_trans.transform(X_raw)

    return X_transformed, distance_km

# ----------------------------------------------------
# 4. API Endpoints
# ----------------------------------------------------
@app.get("/")
def home():
    return {
        "status": "success",
        "message": "Causal Inference DML API is running smoothly!",
        "docs_url": "/docs"
    }

@app.post("/predict_causal_effect")
def predict_causal_effect(data: OrderInput):
    try:
        X_trans, T_val = preprocess_input(data)

        response = {
            "treatment_distance_km": float(T_val),
            "predictions": {}
        }

        for outcome, model in dml_models.items():
            effect = model.effect(X_trans)[0]
            
            response["predictions"][outcome] = {
                "marginal_effect_per_km": float(effect),
                "total_estimated_treatment_effect": float(effect * T_val)
            }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))