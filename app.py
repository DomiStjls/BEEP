import os
from fastapi import FastAPI, File, UploadFile, Request, HTTPException # type: ignore
from fastapi.responses import HTMLResponse # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from fastapi.templating import Jinja2Templates # type: ignore
from fastapi.exceptions import RequestValidationError # type: ignore
from fastapi.responses import PlainTextResponse # type: ignore
from starlette.exceptions import HTTPException as StarletteHTTPException # type: ignore
from utils import predict_tumor, get_polygon
import shutil
from PIL import Image, ImageDraw # type: ignore

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "static/uploads"

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return templates.TemplateResponse(
        "error.html", {"request": request, "result": None, "filename": None}
    )


@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "result": None, "filename": None}
    )

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse(
        "about.html", {"request": request, "result": None, "filename": None}
        )

@app.post("/", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    polygon = get_polygon(filepath)
    img = Image.open(filepath)
    new_polygon = []
    width, height = img.size
    for i in range(0, len(polygon), 2):
        new_polygon.append((polygon[i] * width, polygon[i + 1] * height))
    drw = ImageDraw.Draw(img, 'RGBA')
    drw.polygon(new_polygon, outline=(255, 128, 128), fill=(255, 128, 128, 200))
    img.save(filepath)
    result = predict_tumor(filepath)
    return templates.TemplateResponse(
        "index.html", {"request": request, "result": result, "filename": file.filename}
    )


