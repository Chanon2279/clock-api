from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
from torchvision import transforms
from .model import ClockMultiLabel

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device (CPU หรือ GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ClockMultiLabel().to(device)
model.load_state_dict(torch.load('app/clock_model.pth', map_location=device))
model.eval()

# Transform (เหมือนตอนเทรน ไม่ใส่ RandomRotation)
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
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = transform(image).unsqueeze(0).to(device).float()

        with torch.no_grad():
            output = model(image)
            digit_prob, hand_prob = output[0][0].item(), output[0][1].item()

        return {
            "digit_score": 1 if digit_prob >= 0.4 else 0,
            "hand_score": 1 if hand_prob >= 0.4 else 0
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
