import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
import io
from PIL import Image
from torchvision import transforms
import csv
import os
from datetime import datetime
import joblib

# NEW: Import Transformers for your BERT model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import your Image Model Architecture
from Img_model import MyCNNModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
IMAGE_MODEL_PATH = "plant_disease_model_final.pth"
TEXT_MODEL_PATH = "./plant_bert_model"   # <--- Points to your new BERT folder
LABEL_ENCODER_PATH = "label_encoder.pkl" # <--- Needed to decode text predictions
HISTORY_FILE = "scan_history.csv"

# --- 1. LOAD IMAGE MODEL ---
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

device = torch.device("cpu")
img_model = MyCNNModel(num_classes=len(CLASS_NAMES))

try:
    img_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
    img_model.to(device)
    img_model.eval()
    print("✅ Image Model Loaded!")
except Exception as e:
    print(f"❌ Error Image Model: {e}")

# --- 2. LOAD TEXT MODEL (BERT) ---
text_model = None
tokenizer = None
label_encoder = None

try:
    if os.path.exists(TEXT_MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
        print("⏳ Loading BERT Text Model... (This might take a moment)")
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
        text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
        text_model.eval() # Set to eval mode
        
        # Load the encoder to convert numbers back to "Healthy"/"Disease"
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        print("✅ BERT Text Model Loaded!")
    else:
        print("⚠️ BERT Model folder or label_encoder.pkl not found.")
except Exception as e:
    print(f"❌ Error Text Model: {e}")


# --- HELPER: Image Transform ---
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0).to(device)


# --- API ENDPOINT ---
@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...), 
    description: str = Form("")
):
    try:
        # ==========================
        # 1. IMAGE PREDICTION
        # ==========================
        img_contents = await file.read()
        img_tensor = transform_image(img_contents)
        
        with torch.no_grad():
            output = img_model(img_tensor)
            probs = F.softmax(output, dim=1)
            confidence, pred_idx = probs.max(1)
        
        img_class = CLASS_NAMES[pred_idx.item()]
        img_conf = confidence.item() * 100
        
        # Simple logic to determine if image thinks it's healthy
        img_status = "Healthy" if "healthy" in img_class.lower() else "Disease"

        # ==========================
        # 2. TEXT PREDICTION (BERT)
        # ==========================
        text_result = "Not Provided"
        text_conf = 0.0
        
        if text_model and tokenizer and description.strip():
            # Tokenize
            inputs = tokenizer(description, return_tensors="pt", truncation=True, padding=True, max_length=128)
            
            with torch.no_grad():
                outputs = text_model(**inputs)
                # Get probabilities
                probs = F.softmax(outputs.logits, dim=1)
                conf, pred_id = probs.max(1)
            
            # Decode label (0/1 -> "Healthy"/"Disease")
            text_result = label_encoder.inverse_transform([pred_id.item()])[0]
            text_conf = conf.item() * 100

        # ==========================
        # 3. SAVE HISTORY & RETURN
        # ==========================
        # Save to CSV
        if not os.path.exists(HISTORY_FILE):
             with open(HISTORY_FILE, mode='w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(["Timestamp", "Image", "Description", "Image_Pred", "Text_Pred"])

        with open(HISTORY_FILE, mode='a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                file.filename,
                description,
                img_class,
                text_result
            ])

        return {
            "image": {
                "diagnosis": img_class,
                "status": img_status,
                "confidence": f"{img_conf:.2f}%"
            },
            "text": {
                "diagnosis": text_result,
                "confidence": f"{text_conf:.2f}%"
            }
        }

    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)