import json
import os


def export_json(result: dict, output_dir: str, filename: str, page_num: int) -> str:
    """
    Сохраняет document_json в .json файл, возвращает путь к файлу.
    """
    doc_json = result.get("document_json")
    if not doc_json:
        return None

    os.makedirs(output_dir, exist_ok=True)
    doc_name = os.path.splitext(filename)[0]
    out_path = os.path.join(output_dir, f"{doc_name}_page{page_num + 1}_doc.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc_json, f, ensure_ascii=False, indent=2)

    return out_path


def export_items(result: dict, output_dir: str, filename: str, page_num: int) -> str:
    """
    Сохраняет table_items в .json файл, возвращает путь к файлу.
    """
    table_items = result.get("table_items")
    if not table_items or len(table_items) == 0:
        return None

    os.makedirs(output_dir, exist_ok=True)
    doc_name = os.path.splitext(filename)[0]
    out_path = os.path.join(output_dir, f"{doc_name}_page{page_num + 1}_items.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(table_items, f, ensure_ascii=False, indent=2)

    return out_path


def export_all(result: dict, output_dir: str, filename: str, page_num: int) -> dict:
    """
    Сохраняет и document_json и table_items, возвращает словарь с путями.
    """
    return {
        "json_path": export_json(result, output_dir, filename, page_num),
        "items_path": export_items(result, output_dir, filename, page_num),
    }


def result_to_items_bytes(result: dict) -> bytes:
    """
    Конвертирует table_items в байты JSON — для отдачи через API или Streamlit
    без сохранения на диск.
    """
    table_items = result.get("table_items")
    if not table_items or len(table_items) == 0:
        return None

    return json.dumps(table_items, ensure_ascii=False, indent=2).encode("utf-8")