import json
import os
from io import StringIO

import pandas as pd


def export_json(result: dict, output_dir: str, filename: str, page_num: int) -> str:
    """
    Сохраняет document_json в .json файл, возвращает путь к файлу.
    """
    doc_json = result.get("document_json")
    if not doc_json:
        return None

    os.makedirs(output_dir, exist_ok=True)
    doc_name = os.path.splitext(filename)[0]
    out_path = os.path.join(output_dir, f"{doc_name}_page{page_num + 1}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc_json, f, ensure_ascii=False, indent=2)

    return out_path


def export_csv(result: dict, output_dir: str, filename: str, page_num: int) -> str:
    """
    Сохраняет table_csv в .csv файл, возвращает путь к файлу.
    """
    table_data = result.get("table_csv")
    if not table_data or len(table_data) == 0:
        return None

    os.makedirs(output_dir, exist_ok=True)
    doc_name = os.path.splitext(filename)[0]
    out_path = os.path.join(output_dir, f"{doc_name}_page{page_num + 1}.csv")

    df = pd.DataFrame(table_data)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    return out_path


def export_all(result: dict, output_dir: str, filename: str, page_num: int) -> dict:
    """
    Сохраняет и JSON и CSV, возвращает словарь с путями к файлам.
    """
    return {
        "json_path": export_json(result, output_dir, filename, page_num),
        "csv_path": export_csv(result, output_dir, filename, page_num),
    }


def result_to_csv_bytes(result: dict) -> bytes:
    """
    Конвертирует table_csv из результата в байты CSV — для отдачи через API или Streamlit
    без сохранения на диск.
    """
    table_data = result.get("table_csv")
    if not table_data or len(table_data) == 0:
        return None

    df = pd.DataFrame(table_data)
    buffer = StringIO()
    df.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue().encode("utf-8-sig")