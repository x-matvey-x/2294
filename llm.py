import os
from dotenv import load_dotenv

import base64
import io
import requests
import json
import os
from PIL import Image
from openai import OpenAI


def encode_img_base64(image_path):
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format='JPEG')
        return base64.b64encode(buffered.getvalue()).decode()



def qwen_ocr(image_path, filename, page_num, total_pages, api_key):
    try:
        img_base64 = encode_img_base64(image_path)
        
        prompt = """Проанализируй изображение документа и извлеки информацию В ДВА БЛОКА.

БЛОК 1 - JSON с информацией о документе (все реквизиты):
{
  "Номер_счета": "1966/дА-23",
  "Дата_выставления": "07.08.2023",
  "Реквизиты": {
    "Банк_получателя": "ПАО \\"Промсвязьбанк\\" г. Москва",
    "БИК": "044525555",
    "Корреспондентский_счет": "30101810400000000555",
    "Получатель": "ООО \\"МКВ\\"",
    "ИНН": "7727613771",
    "КПП": "773101001",
    "Расчетный_счет": "40702810600000273395"
  },
  "Исполнитель": {
    "Имя": "ООО \\"МКВ\\"",
    "ИНН": "7727613771",
    "КПП": "773101001",
    "Адрес": "121359, Москва, г. Маршала Тимошенко ул, дом № 44, этаж 1, помещение 1, комната 16",
    "Телефон": "+7 (495) 640-22-00"
  },
  "Заказчик": {
    "Имя": "Национальный исследовательский университет \\"Высшая школа экономики\\", НИУ \\"ВШЭ\\", ВШЭ",
    "ИНН": "7714030276",
    "КПП": "770101001",
    "Адрес": "101000, Город Москва, ул. Мясницкая, дом 20"
  },
  "Основание": {
    "Договор": "№Д/АРМИЯ-2023/612/МД",
    "Дата": "14.07.2023"
  }
}

БЛОК 2 - CSV таблица (массив объектов для преобразования в CSV):
[
  {
    "столбец_1": "значение",
    "столбец_2": "значение"
  }
]

ИТОГОВАЯ СТРУКТУРА ОТВЕТА:
{
  "document_json": { /* JSON с реквизитами */ },
  "table_csv": [ /* массив строк таблицы */ ],
  "totals": {
    "Итого": 192000.00,
    "НДС_20": 32000.00,
    "Всего_к_оплате": 192000.00
  }
}

ПРАВИЛА ДЛЯ ТАБЛИЦЫ:
1. Извлекай ТОЧНЫЕ названия столбцов из заголовка таблицы
2. Если столбец называется "№" - используй "№"
3. Если "Товары (работы, услуги)" - используй "Товары (работы, услуги)"
4. Каждая строка таблицы = один объект в массиве
5. Все ЧИСЛОВЫЕ значения (цены, суммы, количество) - в формате float БЕЗ пробелов
   Правильно: 192000.00
   Неправильно: "192 000,00"

ПРАВИЛА ДЛЯ РЕКВИЗИТОВ:
1. Если блок отсутствует - верни null
2. Даты - СТРОГО "ДД.ММ.ГГГГ"
3. ИНН, КПП, БИК, счета - ТОЛЬКО цифры без пробелов
4. Названия организаций - ТОЧНО как написано с кавычками

СПЕЦИАЛЬНЫЕ СЛУЧАИ:
- Только таблица без реквизитов: "document_json": null
- Только реквизиты без таблицы: "table_csv": null, "totals": null
- Рукописный текст ИГНОРИРУЙ

Верни ТОЛЬКО JSON без markdown разметки."""

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "qwen/qwen-2.5-vl-72b-instruct",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }],
            "temperature": 0.1,
            "max_tokens": 4096
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"Ошибка API {response.status_code}: {response.text}")
        
        result_text = response.json()["choices"][0]["message"]["content"].strip()
        
        ocr_result = json.loads(result_text)
        
        # Метаданные
        ocr_result["metadata"] = {
            "document_name": os.path.splitext(filename)[0],
            "page": page_num + 1,
            "total_pages": total_pages
        }
        
        return ocr_result
        
    except json.JSONDecodeError as e:
        return {
            "document_json": None,
            "table_csv": None,
            "totals": None,
            "metadata": {
                "error": f"Ошибка парсинга: {str(e)}",
                "raw_response": result_text[:500]
            }
        }
    except Exception as e:
        return {
            "document_json": None,
            "table_csv": None,
            "totals": None,
            "metadata": {"error": str(e)}
        }



def local_ocr(image_path, filename, page_num, total_pages):
    try:
        img_base64 = encode_img_base64(image_path)

        prompt = """Проанализируй изображение документа и извлеки информацию В ДВА БЛОКА.

БЛОК 1 - JSON с информацией о документе (все реквизиты):
{
  "Номер_счета": "1966/дА-23",
  "Дата_выставления": "07.08.2023",
  "Реквизиты": {
    "Банк_получателя": "ПАО \\"Промсвязьбанк\\" г. Москва",
    "БИК": "044525555",
    "Корреспондентский_счет": "30101810400000000555",
    "Получатель": "ООО \\"МКВ\\"",
    "ИНН": "7727613771",
    "КПП": "773101001",
    "Расчетный_счет": "40702810600000273395"
  },
  "Исполнитель": {
    "Имя": "ООО \\"МКВ\\"",
    "ИНН": "7727613771",
    "КПП": "773101001",
    "Адрес": "121359, Москва, г. Маршала Тимошенко ул, дом № 44, этаж 1, помещение 1, комната 16",
    "Телефон": "+7 (495) 640-22-00"
  },
  "Заказчик": {
    "Имя": "Национальный исследовательский университет \\"Высшая школа экономики\\", НИУ \\"ВШЭ\\", ВШЭ",
    "ИНН": "7714030276",
    "КПП": "770101001",
    "Адрес": "101000, Город Москва, ул. Мясницкая, дом 20"
  },
  "Основание": {
    "Договор": "№Д/АРМИЯ-2023/612/МД",
    "Дата": "14.07.2023"
  }
}

БЛОК 2 - CSV таблица (массив объектов для преобразования в CSV):
[
  {
    "столбец_1": "значение",
    "столбец_2": "значение"
  }
]

ИТОГОВАЯ СТРУКТУРА ОТВЕТА:
{
  "document_json": { /* JSON с реквизитами */ },
  "table_csv": [ /* массив строк таблицы */ ],
  "totals": {
    "Итого": 192000.00,
    "НДС_20": 32000.00,
    "Всего_к_оплате": 192000.00
  }
}

ПРАВИЛА ДЛЯ ТАБЛИЦЫ:
1. Извлекай ТОЧНЫЕ названия столбцов из заголовка таблицы
2. Если столбец называется "№" - используй "№"
3. Если "Товары (работы, услуги)" - используй "Товары (работы, услуги)"
4. Каждая строка таблицы = один объект в массиве
5. Все ЧИСЛОВЫЕ значения (цены, суммы, количество) - в формате float БЕЗ пробелов
   Правильно: 192000.00
   Неправильно: "192 000,00"

ПРАВИЛА ДЛЯ РЕКВИЗИТОВ:
1. Если блок отсутствует - верни null
2. Даты - СТРОГО "ДД.ММ.ГГГГ"
3. ИНН, КПП, БИК, счета - ТОЛЬКО цифры без пробелов
4. Названия организаций - ТОЧНО как написано с кавычками

СПЕЦИАЛЬНЫЕ СЛУЧАИ:
- Только таблица без реквизитов: "document_json": null
- Только реквизиты без таблицы: "table_csv": null, "totals": null
- Рукописный текст ИГНОРИРУЙ

Верни ТОЛЬКО JSON без markdown разметки."""

        client = OpenAI(
            api_key="EMPTY",
            base_url="http://172.18.146.27:8009/v1",
            timeout=3600
        )

        response = client.chat.completions.create(
            model="tencent/HunyuanOCR",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }],
            temperature=0.0,
            max_tokens=2000
        )

        result_text = response.choices[0].message.content.strip()

        ocr_result = json.loads(result_text)
        ocr_result["metadata"] = {
            "document_name": os.path.splitext(filename)[0],
            "page": page_num + 1,
            "total_pages": total_pages
        }

        return ocr_result

    except json.JSONDecodeError as e:
        return {
            "document_json": None,
            "table_csv": None,
"totals": None,
            "metadata": {"error": f"Ошибка парсинга: {str(e)}", "raw_response": result_text[:500]}
        }
    except Exception as e:
        return {
            "document_json": None,
            "table_csv": None,
            "totals": None,
            "metadata": {"error": str(e)}
        }
