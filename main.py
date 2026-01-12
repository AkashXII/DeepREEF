from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import joblib
from PIL import Image
import io

# ----------------------------------------------------
# FastAPI setup
# ----------------------------------------------------
app = FastAPI(title="DeepReef AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Device
# ----------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------
# Load ResNet50 model
# ----------------------------------------------------
MODEL_PATH = "resnet50_coral.pth"
NUM_CLASSES = 2

model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

CLASS_LABELS = ["Bleached Coral", "Healthy Coral"]

# ----------------------------------------------------
# Load XGBoost model + encoder + columns
# ----------------------------------------------------
rf = joblib.load("xgboost_env_severity_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
X_train_columns = joblib.load("train_columns-2.pkl")

# ----------------------------------------------------
# Handle encoder safely (dict or LabelEncoder)
# ----------------------------------------------------
index_to_label = {}

if isinstance(label_encoder, dict):
    # label -> index  OR  index -> label
    sample_value = next(iter(label_encoder.values()))
    if isinstance(sample_value, int):
        # label -> index  → invert
        index_to_label = {v: k for k, v in label_encoder.items()}
    else:
        # already index -> label
        index_to_label = label_encoder
else:
    # sklearn LabelEncoder
    for i, label in enumerate(label_encoder.classes_):
        index_to_label[i] = label

# ----------------------------------------------------
# Image preprocessing (ResNet50)
# ----------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(uploaded_file: UploadFile):
    img_bytes = uploaded_file.file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = transform(img).unsqueeze(0)
    return img.to(DEVICE)

# ----------------------------------------------------
# Main endpoint
# ----------------------------------------------------
@app.post("/analyze_coral")
async def analyze_coral(
    file: UploadFile = File(...),
    Temperature_Mean: float = Form(...),
    Windspeed: float = Form(...),
    TSA: float = Form(...),
    Ocean_Name: str = Form(...),
    Exposure: str = Form(...)
):
    # -------- Image inference --------
    img_tensor = preprocess_image(file)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0][idx].item())

    coral_status = CLASS_LABELS[idx]

    # -------- Normalize categorical inputs --------
    ocean = Ocean_Name.strip().lower()
    exposure = Exposure.strip().lower()

    if "pacific" in ocean:
        ocean = "Pacific"
    elif "atlantic" in ocean:
        ocean = "Atlantic"
    else:
        ocean = "Unknown"

    if "sheltered" in exposure:
        exposure = "Sheltered"
    elif "exposed" in exposure:
        exposure = "Exposed"
    else:
        exposure = "Unknown"

    # -------- XGBoost inference --------
    df = pd.DataFrame([{
        "Temperature_Mean": Temperature_Mean,
        "Windspeed": Windspeed,
        "TSA": TSA,
        "Ocean_Name": ocean,
        "Exposure": exposure
    }])

    df_enc = pd.get_dummies(df, columns=["Ocean_Name", "Exposure"], drop_first=True)

    for col in X_train_columns:
        if col not in df_enc.columns:
            df_enc[col] = 0

    df_enc = df_enc[X_train_columns]

    severity_pred = rf.predict(df_enc)
    severity_index = int(severity_pred[0])
    severity = index_to_label.get(severity_index, "Unknown")

    return {
        "image_prediction": coral_status,
        "confidence": round(confidence * 100, 2),
        "bleaching_severity": severity
    }

# ----------------------------------------------------
# Root endpoint
# ----------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "DeepReef AI backend running",
        "endpoint": "/analyze_coral"
    }
