from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
from torchvision import transforms
from .model import ClockMultiLabel  # Import model

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
model = ClockMultiLabel().to(device)
model.load_state_dict(torch.load('app/clock_model.pth', map_location=device))
model.eval()

# Transform (ใช้เหมือนตอนเทรน)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

@app.get("/")
def root():
    return {"message": "Clock API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # อ่านไฟล์รูป
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = transform(image).unsqueeze(0).to(device).float()

        # Predict
        with torch.no_grad():
            output = model(image)
            probs = torch.sigmoid(output)[0]  # Apply Sigmoid HERE!
            digit_prob, hand_prob = probs[0].item(), probs[1].item()

        return {
            "digit_score": 1 if digit_prob >= 0.5 else 0,
            "digit_prob": round(digit_prob, 3),
            "hand_score": 1 if hand_prob >= 0.5 else 0,
            "hand_prob": round(hand_prob, 3)
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})