from fastapi import FastAPI, File, UploadFile
from pathlib import Path
import sys

# Locate the Streamlit application folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STREAMLIT_APP = PROJECT_ROOT / "streamlit_app"

sys.path.insert(0, str(STREAMLIT_APP))

from model_helper import predict

app = FastAPI()


@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        image_path = Path(__file__).parent / "temp_file.jpg"

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        prediction = predict(str(image_path))

        return {"prediction": prediction}

    except Exception as e:
        return {"error": str(e)}