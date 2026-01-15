from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pickle
import numpy as np
import os
from datetime import datetime

# ========== Data Model ==========
class PatientData(BaseModel):
    # Basic info
    Age: int
    Gender: str  # "Male" or "Female"
    
    # Vital signs
    Systolic_BP: int
    Diastolic_BP: int
    Body_Temperature_Celsius: float
    Heart_Rate_bpm: int
    Blood_Sugar_mg_dL: int
    BMI: float
    
    # Symptoms (0 or 1 for each)
    symptom_body_pain: int = 0
    symptom_dry_mouth: int = 0
    symptom_fainting: int = 0
    symptom_fatigue: int = 0
    symptom_fever: int = 0
    symptom_frequent_urination: int = 0
    symptom_headache: int = 0
    symptom_low_activity: int = 0
    symptom_thirst: int = 0
    symptom_weakness: int = 0
    symptom_weight_loss: int = 0

# ========== Initialize FastAPI ==========
app = FastAPI(
    title="Medical Diagnosis Predictor API",
    description="Simple ML model for disease prediction",
    version="1.0.0"
)

# Allow all origins for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Global ML Model ==========
model = None
expected_features = None

# ========== Load ML Model ==========
def load_ml_model():
    """Load the ML model on startup"""
    global model, expected_features
    
    try:
        print("🚀 Loading ML model...")
        
        # Load the model
        with open('ml_model/model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Check if model has feature names attribute
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
            print(f"✅ Model loaded with {len(expected_features)} features:")
            for i, feat in enumerate(expected_features):
                print(f"   {i+1}. {feat}")
        else:
            print("⚠️ Model doesn't have feature names attribute")
            # Create expected features list based on your input format
            expected_features = [
                'Age', 'Systolic_BP', 'Diastolic_BP', 'Body_Temperature_Celsius',
                'Heart_Rate_bpm', 'Blood_Sugar_mg_dL', 'BMI', 'Gender_Male',
                'symptom_body_pain', 'symptom_dry_mouth', 'symptom_fainting',
                'symptom_fatigue', 'symptom_fever', 'symptom_frequent_urination',
                'symptom_headache', 'symptom_low_activity', 'symptom_thirst',
                'symptom_weakness', 'symptom_weight_loss'
            ]
        
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise e

# ========== Feature Preparation ==========
def prepare_features(patient_data: PatientData):
    """Convert patient data to feature array expected by model"""
    
    # Convert gender to binary
    gender_male = 1 if patient_data.Gender.lower() == 'male' else 0
    
    # Create feature array in EXACT order expected by model
    features = np.array([[
        patient_data.Age,
        patient_data.Systolic_BP,
        patient_data.Diastolic_BP,
        patient_data.Body_Temperature_Celsius,
        patient_data.Heart_Rate_bpm,
        patient_data.Blood_Sugar_mg_dL,
        patient_data.BMI,
        gender_male,
        patient_data.symptom_body_pain,
        patient_data.symptom_dry_mouth,
        patient_data.symptom_fainting,
        patient_data.symptom_fatigue,
        patient_data.symptom_fever,
        patient_data.symptom_frequent_urination,
        patient_data.symptom_headache,
        patient_data.symptom_low_activity,
        patient_data.symptom_thirst,
        patient_data.symptom_weakness,
        patient_data.symptom_weight_loss
    ]])
    
    return features

# ========== Risk Level Calculation ==========
def get_risk_level(disease: str, confidence: float, age: int, systolic_bp: int) -> str:
    """Determine risk level"""
    
    # High risk conditions
    if 'hypertension' in disease.lower() and systolic_bp > 160:
        return 'High'
    elif 'diabetes' in disease.lower() and confidence > 0.7:
        return 'High'
    elif age > 60 and confidence > 0.8:
        return 'High'
    elif confidence > 0.85:
        return 'Medium'
    else:
        return 'Low'

# ========== Recommendations ==========
def get_recommendations(disease: str, risk_level: str) -> List[str]:
    """Generate recommendations"""
    
    base_recommendations = {
        'Hypertension': [
            "Monitor blood pressure daily",
            "Reduce sodium intake",
            "Consult cardiologist if BP > 140/90",
            "Regular exercise"
        ],
        'Diabetes': [
            "Monitor blood sugar levels",
            "Follow diabetic diet",
            "Regular exercise",
            "Consult endocrinologist"
        ],
        'Fever': [
            "Rest and hydrate",
            "Monitor temperature",
            "Consult if fever > 3 days"
        ],
        'default': [
            "Consult healthcare provider",
            "Follow up if symptoms worsen"
        ]
    }
    
    # Find matching disease
    disease_lower = disease.lower()
    recommendations = base_recommendations['default']
    
    for key in base_recommendations:
        if key.lower() in disease_lower:
            recommendations = base_recommendations[key]
            break
    
    # Add risk-specific advice
    if risk_level == 'High':
        recommendations.insert(0, "Seek medical attention soon")
    elif risk_level == 'Medium':
        recommendations.insert(0, "Schedule doctor appointment")
    
    return recommendations

# ========== API Endpoints ==========
@app.on_event("startup")
async def startup_event():
    """Load ML model when server starts"""
    load_ml_model()

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "healthy",
        "service": "Medical Diagnosis API",
        "model_loaded": model is not None,
        "features_expected": expected_features,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/features")
async def get_expected_features():
    """Get the features expected by the model"""
    if expected_features:
        return {
            "features": expected_features,
            "count": len(expected_features),
            "feature_order": expected_features
        }
    return {"error": "Feature names not available"}

@app.get("/api/model-info")
async def get_model_info():
    """Get model information"""
    if model is None:
        return {"error": "Model not loaded"}
    
    info = {
        "model_type": type(model).__name__,
        "model_loaded": True,
        "n_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else "unknown",
        "n_classes": len(model.classes_) if hasattr(model, 'classes_') else "unknown",
        "classes": model.classes_.tolist() if hasattr(model, 'classes_') else []
    }
    
    return info

@app.post("/api/predict")
async def predict(patient_data: PatientData):
    """Make a prediction"""
    start_time = datetime.now()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please wait.")
    
    try:
        # Prepare features
        features = prepare_features(patient_data)
        
        # Make prediction
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)
        
        # Get results
        disease_name = prediction[0]
        confidence = float(np.max(probabilities[0]))
        
        # Get top 3 predictions
        top_n = 3
        sorted_indices = np.argsort(probabilities[0])[::-1][:top_n]
        
        top_predictions = []
        for idx in sorted_indices:
            disease = model.classes_[idx] if hasattr(model, 'classes_') else f"Class_{idx}"
            prob = float(probabilities[0][idx])
            top_predictions.append({
                "disease": disease,
                "probability": prob,
                "percentage": round(prob * 100, 1)
            })
        
        # Determine risk level
        risk_level = get_risk_level(
            disease_name, 
            confidence, 
            patient_data.Age,
            patient_data.Systolic_BP
        )
        
        # Get recommendations
        recommendations = get_recommendations(disease_name, risk_level)
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time_ms = round((end_time - start_time).total_seconds() * 1000, 2)
        
        return {
            "disease": disease_name,
            "confidence": confidence,
            "confidence_percentage": round(confidence * 100, 1),
            "risk_level": risk_level,
            "recommendations": recommendations,
            "top_predictions": top_predictions,
            "processing_time_ms": processing_time_ms,
            "features_used": len(features[0]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Simple test endpoint
@app.post("/api/test")
async def test_prediction():
    """Test endpoint with sample data"""
    test_data = {
        "Age": 45,
        "Gender": "Male",
        "Systolic_BP": 140,
        "Diastolic_BP": 90,
        "Body_Temperature_Celsius": 36.6,
        "Heart_Rate_bpm": 72,
        "Blood_Sugar_mg_dL": 110,
        "BMI": 24.5,
        "symptom_body_pain": 0,
        "symptom_dry_mouth": 0,
        "symptom_fainting": 0,
        "symptom_fatigue": 1,
        "symptom_fever": 0,
        "symptom_frequent_urination": 0,
        "symptom_headache": 1,
        "symptom_low_activity": 0,
        "symptom_thirst": 0,
        "symptom_weakness": 0,
        "symptom_weight_loss": 0
    }
    
    patient_data = PatientData(**test_data)
    return await predict(patient_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )