import os

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.responses import JSONResponse
import traceback
from pipeline.pipeline import run
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="VLM OCR API",
    description="OCR сервис на основе Vision Language Model",
    version="1.0.0",
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"error": str(exc)})

@app.get("/health")
def health():
    """
    Проверка что сервер живой
    """
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    use_local: bool = Query(default=True, description="True — локальная модель, False — Qwen через OpenRouter"),
):
    """
    Принимает файл (PDF, ZIP, PNG, JPG), возвращает результат OCR.
    """
    filename = file.filename
    allowed_extensions = (".pdf", ".zip", ".png", ".jpg", ".jpeg")

    if not filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат файла. Допустимые: {', '.join(allowed_extensions)}",
        )

    file_bytes = await file.read()

    api_key = os.getenv("OPENROUTER_API_KEY") if not use_local else None

    if not use_local and not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY не задан в переменных окружения",
        )

    save_dir = "saved_images"
    os.makedirs(save_dir, exist_ok=True)

    results = run(
        file_bytes=file_bytes,
        filename=filename,
        save_dir=save_dir,
        output_dir=None,
        use_local=use_local,
        api_key=api_key,
    )

    return JSONResponse(content={"results": results})
