from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
from torchvision import transforms
from .model import ClockClassifier  # Import model

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
model.load_state_dict(torch.load('app/clock_model_multiclass.pth', map_location=device))
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
            probs = torch.softmax(output, dim=1)[0]  # Apply Softmax to get probabilities
            pred_class = torch.argmax(probs).item()  # Get predicted class

        # Map class to label (digit, hand) pair
        class_map = {
            0: (1, 1),
            1: (1, 0),
            2: (0, 1),
            3: (0, 0)
        }
        digit_score, hand_score = class_map[pred_class]
        digit_prob, hand_prob = probs[pred_class].item(), probs[pred_class].item()

        return {
            "digit_score": digit_score,
            "digit_prob": round(digit_prob, 3),
            "hand_score": hand_score,
            "hand_prob": round(hand_prob, 3)
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
