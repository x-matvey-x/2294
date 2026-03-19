import json


def parse_response(raw_text: str) -> dict:
    """
    Парсит сырой текст ответа модели в словарь.
    Если модель вернула невалидный JSON — возвращает словарь с ошибкой.
    """
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        return {
            "document_json": None,
            "table_csv": None,
            "totals": None,
            "error": f"Ошибка парсинга JSON: {str(e)}",
            "raw_response": raw_text[:500],
        }


def attach_metadata(result: dict, filename: str, page_num: int, total_pages: int) -> dict:
    """
    Добавляет метаданные к результату — имя документа, номер страницы.
    """
    result["metadata"] = {
        "document_name": filename,
        "page": page_num + 1,
        "total_pages": total_pages,
    }
    return result


def process_result(raw_text: str, filename: str, page_num: int, total_pages: int) -> dict:
    """
    Главная функция — парсит ответ модели и добавляет метаданные.
    """
    result = parse_response(raw_text)
    result = attach_metadata(result, filename, page_num, total_pages)
    return result