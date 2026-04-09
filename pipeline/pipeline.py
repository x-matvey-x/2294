import os
from pipeline.document_loader import load_document
from pipeline.vlm_inference import run_qwen, run_local
from pipeline.postprocessing import process_result
from pipeline.export import export_all


def run_page(image_path: str, filename: str, page_num: int, total_pages: int, use_local: bool = True, api_key: str = None) -> dict:
    """
    Обрабатывает одну страницу документа.
    Возвращает словарь с результатом OCR.
    """
    if use_local:
        raw_text = run_local(image_path)
    else:
        if not api_key:
            raise ValueError("api_key обязателен при use_local=False")
        raw_text = run_qwen(image_path, api_key)

    result = process_result(raw_text, filename, page_num, total_pages)
    return result


def run(file_bytes: bytes, filename: str, save_dir: str, output_dir: str = None, use_local: bool = True, api_key: str = None) -> list[dict]:
    """
    Главная функция пайплайна.
    Принимает файл, прогоняет все страницы через OCR, возвращает список результатов.
    Если передан output_dir — сохраняет JSON и CSV на диск.
    """
    image_paths = load_document(file_bytes, filename, save_dir)
    total_pages = len(image_paths)
    results = []

    for page_num, image_path in enumerate(image_paths):
        print(f"Обрабатываю страницу {page_num + 1} из {total_pages}...")

        result = run_page(
            image_path=image_path,
            filename=filename,
            page_num=page_num,
            total_pages=total_pages,
            use_local=use_local,
            api_key=api_key,
        )

        if output_dir:
            paths = export_all(result, output_dir, filename, page_num)
            result["exported"] = paths

        results.append(result)

    return results

def run_batch(zip_bytes: bytes, save_dir: str, output_dir: str = None, use_local: bool = True, api_key: str = None) -> dict:
    """
    Принимает ZIP с PDF документами, обрабатывает каждый.
    Возвращает словарь: имя файла → список результатов по страницам.
    """
    from pipeline.document_loader import unzip_documents
    
    documents = unzip_documents(zip_bytes)
    batch_results = {}

    for filename, file_bytes in documents:
        print(f"\nДокумент: {filename}")
        results = run(
            file_bytes=file_bytes,
            filename=filename,
            save_dir=save_dir,
            output_dir=output_dir,
            use_local=use_local,
            api_key=api_key,
        )
        batch_results[filename] = results

    return batch_results