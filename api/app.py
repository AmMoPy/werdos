from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastai.vision.all import load_learner
from io import BytesIO
from PIL import Image

# Load the model
learn = load_learner('/cnn/model_2_ep_new_data.pkl')

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ammopy.github.io/werdos"],
    allow_methods=["POST"],  # Allow only POST requests
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(image: UploadFile = File(..., description = "Upload an image for classification")):
    try:
        # Validate file type
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code = 400, detail = "Only images are allowed")

        # Read and preprocess the image
        img_bytes = await image.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        # Make predictions
        preds, idx, probs = learn.predict(img)
        cats = ['Dog', 'Cat']
        result = dict(zip(cats, map(float, probs)))

        return {"success": True, "predictions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))