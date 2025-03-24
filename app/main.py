from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
import numpy as np
from torchvision import transforms
from .model import ClockClassifier  # Make sure this is the correct model import

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

# Transform (like during training, no RandomRotation)
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
        # Read image file and transform
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = transform(image).unsqueeze(0).to(device).float()

        # Make prediction with model
        with torch.no_grad():
            output = model(image)  # Model output is a tensor of shape [1, 2]

            # Extract the predicted class by getting the index of max value
            pred_class = torch.argmax(output, dim=1).item()

            # Map the predicted class to digit/hand score using the class map
            result = label_map.get(pred_class, {"digit_score": 0, "hand_score": 0})

        # Return the predicted results
        return {
            "digit_score": result["digit_score"],
            "hand_score": result["hand_score"]
        }

    except Exception as e:
        # Return detailed error message
        return JSONResponse(status_code=400, content={"error": str(e)})
