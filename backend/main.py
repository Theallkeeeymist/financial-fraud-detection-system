from fastapi import FastAPI, UploadFile, File
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score,classification_report
from numpy.version import version
from pydantic import BaseModel
from typing import List

#initialize FastAPI app
app=FastAPI(
    title="Financial Fraud Detection System",
    description="API for detecting transactional frauds using XGBoost",
    version="1.0.0"
)
print("FastAPI app starting...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "model", "final_xgboost_smote_calibrated_model.pkl")
model=joblib.load(model_path)

print("Model loaded successfully:", model)

class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/")
def ping():
    return {"message": "Fraud Detection API is Live"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": model_path if 'model_path' in locals() else "Not found"
    }

print("Registering /predict endpoint")


@app.post("/predict")
def predict(data: Transaction):
    input_data = np.array([list(data.dict().values())])
    probabilities = model.predict_proba(input_data)[0]  # Get both probabilities
    fraud_prob = probabilities[1]  # Probability of class 1 (Fraud)

    # Use threshold of 0.3
    prediction = "Fraud" if fraud_prob > 0.3 else "Legit"

    return {
        "Prediction": prediction,
        "Confidence": float(fraud_prob),  # Don't round here
        "Probability_Legit": float(probabilities[0]),  # For debugging
        "Probability_Fraud": float(fraud_prob)
    }


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        # Read CSV with error handling
        try:
            df = pd.read_csv(file.file)
        except Exception as e:
            print(f"error: {e}")
            return {"error": f"CSV parsing error: {str(e)}"}

        # Expected columns - must match EXACTLY what model was trained on
        expected_cols = [
            "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
            "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
            "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
        ]

        # Check columns
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            return {"error": f"Missing columns: {missing_cols}"}

        # Check for NaN values
        if df[expected_cols].isnull().any().any():
            return {"error": "CSV contains missing values"}

        # Debug output
        print("First row values:", df[expected_cols].iloc[0].tolist())
        print("Data shape:", df[expected_cols].shape)

        # Get predictions
        probabilities = model.predict_proba(df[expected_cols])
        fraud_probs = probabilities[:, 1]  # Probability of class 1 (Fraud)
        y_pred=(fraud_probs>0.0001).astype(int)

        # Add to your predict_csv endpoint before predictions
        print("Feature means:", df[expected_cols].mean())
        print("Feature stds:", df[expected_cols].std())

        results = []
        for i, prob in enumerate(fraud_probs):
            results.append({
                "index": i,
                "prediction": "Fraud" if prob > 0.0001 else "Legit",
                "confidence": float(prob),
                "features": {col: float(df[col].iloc[i]) for col in expected_cols[:3]}
                # Show first 3 features for debugging
            })

        if 'Class' in df.columns:
            y_true = df['Class'].astype(int)
            acc = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, output_dict=True)

            return {
                "accuracy": acc,
                "classification_report": report,
                "results": results
            }

        return {"results": results, "note": "Accuracy not computed (no 'Class' column in CSV)"}

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


# @app.post("/predict")
# def predict(data: Transaction):
#     input_data = np.array([list(data.dict().values())])
#     probabilities = model.predict_proba(input_data)[0]  # Get both probabilities
#     fraud_prob = probabilities[1]  # Probability of class 1 (Fraud)
#
#     # Use threshold of 0.3
#     prediction = "Fraud" if fraud_prob > 0.3 else "Legit"
#
#     return {
#         "Prediction": prediction,
#         "Confidence": float(fraud_prob),  # Don't round here
#         "Probability_Legit": float(probabilities[0]),  # For debugging
#         "Probability_Fraud": float(fraud_prob)
#     }
#
# #CSV upload standpoint
# @app.post("/predict_csv")
# async def predict_csv(file: UploadFile = File(...)):
#     df=pd.read_csv(file.file)
#
#     importance=pd.DataFrame({
#         'Feature':df.columns[:-1],
#         'Importance': model.feature_importances_
#     }).sort_values('Importance', ascending=False)
#
#     print("Feature Importance: \n",importance.head(10))
#
#     # # Add debug print
#     # print("Input data description:\n", df.describe())
#
#     expected_cols = [
#         "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
#         "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
#         "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
#     ]
#
#     if not all(col in df.columns for col in expected_cols):
#         return {"error": "CSV is missing required columns"}
#
#     # # predictions=model.predict(df[expected_cols])
#     # probability=model.predict_proba(df[expected_cols])[:,1]
#     # # predictions="Fraud" if probability>0.3 else "Legit"
#
#     probabilities = model.predict_proba(df[expected_cols])
#     fraud_probs = probabilities[:, 1]  # Only fraud probabilities
#
#     print("Fraud probabilities sample:", fraud_probs[:5])  # Debug first 5
#
#     results=[]
#     for i,prob in range(len(df)):
#         results.append({
#             "index": i,
#             "prediction": "Fraud" if prob > 0.3 else "Legit",
#             "confidence": float(prob),
#             "probability_legit": float(probabilities[i][0]),
#             "probability_fraud": float(prob)
#         })
#     return {"results": results}