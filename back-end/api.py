from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse

from pathlib import Path
from PIL import Image
from datetime import datetime
import flux, flux_kontent

app = FastAPI()

FLUX_TYPE = None
FLUX_PIPELINE = None

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/upload")
async def upload(image: UploadFile = File(...)):
    if image.content_type not in ["image/png", "image/jpeg", "image/jpg", "image/webp"]:
        raise HTTPException(status_code = 400, detail = "Only image files (.png, .jpeg, .jpg, .webp) are allowed.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_path = UPLOAD_DIR / f"{timestamp}.png"

    with upload_path.open("wb") as buffer:
        data = await image.read()
        buffer.write(data)

    return {"path": str(upload_path)}

@app.post("/download")
async def download(path: str):
    download_path = OUTPUT_DIR / path
    if not download_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(path = download_path, media_type = "image/png", filename = path)

@app.post("/generate")
async def generate(prompt: str, image_path: str = None, width: int = 1024, height: int = 1024):
    global FLUX_TYPE, FLUX_PIPELINE
    
    FLUX_PIPELINE = flux_kontent.load_model()
    image = Image.open(UPLOAD_DIR / image_path)
    result = flux_kontent.generate_image(FLUX_PIPELINE, image, prompt, width, height)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"{timestamp}.png"
    result.save(output_path)

    return {"path": str(output_path)}
    

    
