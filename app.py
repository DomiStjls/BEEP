import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from utils import predict_tumor
import shutil

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "static/uploads"


@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "result": None, "filename": None}
    )

@app.get("/about", response_class=HTMLResponse)
async def get_form_about(request: Request):
    return templates.TemplateResponse(
        "about.html", {"request": request, "result": None, "filename": None}
    )

@app.post("/", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    if not os.path.exists("static/uploads"):
        os.makedirs("static/uploads")
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_tumor(filepath)
    return templates.TemplateResponse(
        "index.html", {"request": request, "result": result, "filename": file.filename}
    )


