import os
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse 
from fastapi.staticfiles import StaticFiles 
from fastapi.templating import Jinja2Templates 
from fastapi.exceptions import RequestValidationError 
from fastapi.responses import PlainTextResponse 
from starlette.exceptions import HTTPException
from utils import predict_tumor
import shutil

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)  
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return templates.TemplateResponse(
        "error.html", {"request": request, "result": None, "filename": None}
    )


@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "result": None, "filename1": None}
    )

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse(
        "about.html", {"request": request, "result": None, "filename1": None}
        )

@app.post("/", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = predict_tumor(filepath)
    filename2 = file.filename[:-4] + '1.jpg'
    return templates.TemplateResponse(
        "index.html", {"request": request, "result": result, "filename1": file.filename, "filename2": filename2}
    )


