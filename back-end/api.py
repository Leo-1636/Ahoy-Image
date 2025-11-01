from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse

from pathlib import Path
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
    target_type = "FLUX.1-Kontext-dev" if image_path else "FLUX.1-dev"
    if FLUX_TYPE != target_type:
        try:
            if target_type == "FLUX.1-dev":
                FLUX_PIPELINE = flux.load_model()
            else:
                FLUX_PIPELINE = flux_kontent.load_model()

            FLUX_TYPE = target_type
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    if FLUX_TYPE == "FLUX.1-dev":
        result = flux.generate_image(FLUX_PIPELINE, prompt, width, height)
    else:
        result = flux_kontent.generate_image(FLUX_PIPELINE, image_path, prompt, width, height)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"{timestamp}.png"
    result.save(output_path)

    return {"path": str(output_path)}
    

    
