import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse

from pipeline.pipeline import run, run_batch

load_dotenv()

app = FastAPI(
    title="VLM OCR API",
    description="OCR сервис на основе Vision Language Model",
    version="1.0.0",
)

ALLOWED_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg")
SAVE_DIR = "saved_images"


def get_api_key(use_local: bool) -> str | None:
    if use_local:
        return None
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY не задан в переменных окружения",
        )
    return api_key


@app.get("/health")
def health():
    """
    Проверка что сервер живой.
    """
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    use_local: bool = Query(default=True, description="True — локальная модель, False — Qwen через OpenRouter"),
):
    """
    Принимает один файл (PDF, PNG, JPG), возвращает результат OCR по всем страницам.
    """
    if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат. Допустимые: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    api_key = get_api_key(use_local)
    os.makedirs(SAVE_DIR, exist_ok=True)

    results = run(
        file_bytes=file_bytes,
        filename=file.filename,
        save_dir=SAVE_DIR,
        output_dir=None,
        use_local=use_local,
        api_key=api_key,
    )

    return JSONResponse(content={"filename": file.filename, "results": results})


@app.post("/ocr/batch")
async def ocr_batch(
    file: UploadFile = File(...),
    use_local: bool = Query(default=True, description="True — локальная модель, False — Qwen через OpenRouter"),
):
    """
    Принимает ZIP архив с PDF документами, возвращает результаты по каждому документу.
    """
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Батч обработка принимает только ZIP архив с PDF файлами внутри",
        )

    file_bytes = await file.read()
    api_key = get_api_key(use_local)
    os.makedirs(SAVE_DIR, exist_ok=True)

    results = run_batch(
        zip_bytes=file_bytes,
        save_dir=SAVE_DIR,
        output_dir=None,
        use_local=use_local,
        api_key=api_key,
    )

    return JSONResponse(content={"results": results})