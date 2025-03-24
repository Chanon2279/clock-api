from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
import numpy as np
from torchvision import transforms
from .model import ClockClassifier  # Make sure to import the correct model

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ClockClassifier().to(device)
model.load_state_dict(torch.load('app/clock_model_multiclass.pth', map_location=device))
model.eval()

# Transform (same as during training, but no RandomRotation)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Class mapping for the new model output
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
async def predict(file: UploadFile = File(...)):
    try:
        # Read image from the request
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = transform(image).unsqueeze(0).to(device).float()

        # Predict
        with torch.no_grad():
            output = model(image)
            # Apply softmax to convert raw scores to probabilities
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            pred_class = np.argmax(probs)  # Get the class with the highest probability

            # Map the predicted class to digit/hand score using the class map
            result = label_map[pred_class]

        # Return the predicted results
        return {
            "digit_score": result["digit_score"],
            "digit_prob": round(probs[pred_class], 3),
            "hand_score": result["hand_score"],
            "hand_prob": round(probs[pred_class], 3)
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

