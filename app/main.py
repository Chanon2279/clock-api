from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
from torchvision import transforms
from model import ClockClassifier  # Assumes model.py is in the same directory

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ClockClassifier().to(device)
model.load_state_dict(torch.load('clock_model_multiclass.pth', map_location=device))
model.eval()

# Transform (no augmentation for inference)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Class mapping for multi-class output
label_map = {
    0: {"digit_score": 1, "hand_score": 1},
    1: {"digit_score": 1, "hand_score": 0},
    2: {"digit_score": 0, "hand_score": 1},
    3: {"digit_score": 0, "hand_score": 0}
}

@app.get("/")
def root():
    return {"message": "Clock API Multi-Class is running!"}

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    correct_digit: int = Form(...),  # Expected digit (0 or 1)
    correct_hand: int = Form(...)    # Expected hand (0 or 1)
):
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = transform(image).unsqueeze(0).to(device).float()

        # Prediction
        with torch.no_grad():
            output = model(image)
            pred_class = torch.argmax(output, dim=1).item()

        # Get scores from prediction
        predicted = label_map[pred_class]
        digit_score = predicted["digit_score"]
        hand_score = predicted["hand_score"]

        # Compare with provided correct values
        digit_match = 1 if digit_score == correct_digit else 0
        hand_match = 1 if hand_score == correct_hand else 0

        return {
            "digit_score": digit_match,
            "hand_score": hand_match,
            "predicted_digit": digit_score,
            "predicted_hand": hand_score
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})