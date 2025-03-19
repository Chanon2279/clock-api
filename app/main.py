from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import io
from torchvision import transforms
from .model import ClockMultiLabel


app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend access
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = ClockMultiLabel()
model.load_state_dict(torch.load('app/clock_model.pth', map_location=torch.device('cpu')))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
        digit_prob, hand_prob = output[0][0].item(), output[0][1].item()
    
    return {
        "digit_score": 1 if digit_prob >= 0.5 else 0,
        "hand_score": 1 if hand_prob >= 0.5 else 0
    }
