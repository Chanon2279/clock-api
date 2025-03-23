from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
from torchvision import transforms
from .model import ClockMultiOutput  # <-- Import ใหม่

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = ClockMultiOutput(num_digit_classes=10, num_hand_classes=12).to(device)
model.load_state_dict(torch.load('app/clock_model.pth', map_location=device))
model.eval()

# Transform
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
            digit_out, hand_out = model(image)
            digit_pred = torch.argmax(digit_out, dim=1).item()
            hand_pred = torch.argmax(hand_out, dim=1).item()

        return {
            "digit_label": int(digit_pred),
            "hand_label": int(hand_pred),
            "result": f"{digit_pred}.{hand_pred}"  # เช่น 11.10
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
