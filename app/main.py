from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
import numpy as np
from torchvision import transforms
<<<<<<< HEAD
from .model import ClockClassifier  # Make sure this is the correct model import
=======
from .model import ClockMultiOutput  # เปลี่ยน import
>>>>>>> parent of 855b569 (fix model)

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

<<<<<<< HEAD
# Device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ClockClassifier().to(device)
model.load_state_dict(torch.load('app/clock_model_multiclass.pth', map_location=device))
=======
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ClockMultiOutput(num_digit_classes=10, num_hand_classes=12).to(device)
model.load_state_dict(torch.load('app/clock_model.pth', map_location=device))
>>>>>>> parent of 855b569 (fix model)
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
async def predict(
    file: UploadFile = File(...),
    correct_digit: int = Form(...),  
    correct_hand: int = Form(...)   
):
    try:
        # Read image file and transform
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = transform(image).unsqueeze(0).to(device).float()

        # Make prediction with model
        with torch.no_grad():
<<<<<<< HEAD
            output = model(image)  # Model output is a tensor of shape [1, 2]
=======
            digit_out, hand_out = model(image)
            digit_pred = torch.argmax(digit_out, dim=1).item()
            hand_pred = torch.argmax(hand_out, dim=1).item()

        # คะแนน
        digit_score = 1 if digit_pred == correct_digit else 0
        hand_score = 1 if hand_pred == correct_hand else 0
>>>>>>> parent of 855b569 (fix model)

            # Extract the predicted class by getting the index of max value
            pred_class = torch.argmax(output, dim=1).item()

            # Map the predicted class to digit/hand score using the class map
            result = label_map.get(pred_class, {"digit_score": 0, "hand_score": 0})

        # Return the predicted results
        return {
<<<<<<< HEAD
            "digit_score": result["digit_score"],
            "hand_score": result["hand_score"]
=======
            "digit_score": digit_score,
            "hand_score": hand_score
>>>>>>> parent of 855b569 (fix model)
        }

    except Exception as e:
        # Return detailed error message
        return JSONResponse(status_code=400, content={"error": str(e)})
