import base64
import io
import os

import requests
import yaml
from openai import OpenAI
from PIL import Image


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "model_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


PROMPT = """Проанализируй изображение документа и извлеки информацию В ДВА БЛОКА.

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

БЛОК 2 - JSON массив строк таблицы. Каждая строка — объект со строго фиксированными полями:
[
  {
    "item_position": "1",
    "item_desc": "Название товара или услуги",
    "item_qty": "10",
    "item_unit": "шт",
    "item_price": "1 000,00",
    "item_amount": "10 000,00",
    "item_vat_rate": "20%",
    "item_vat_amount": "2 000,00",
    "item_total_with_vat": "12 000,00",
    "item_article": "",
    "item_product_code": "",
    "item_country_name": "",
    "item_country_code": "",
    "item_customs_declaration": "",
    "item_excise": ""
  }
]

ИТОГОВАЯ СТРУКТУРА ОТВЕТА:
{
  "document_json": { /* JSON с реквизитами */ },
  "table_items": [ /* массив строк таблицы */ ],
  "totals": {
    "Итого": "192 000,00",
    "НДС_20": "32 000,00",
    "Всего_к_оплате": "192 000,00"
  }
}

ПРАВИЛА ДЛЯ ТАБЛИЦЫ:
1. Используй СТРОГО фиксированные названия полей: item_position, item_desc, item_qty, item_unit, item_price, item_amount, item_vat_rate, item_vat_amount, item_total_with_vat, item_article, item_product_code, item_country_name, item_country_code, item_customs_declaration, item_excise
2. Если поле в документе отсутствует — оставь пустую строку ""
3. Числовые значения оставляй как строки, точно как написано в документе
4. Каждая строка таблицы = один объект в массиве

ПРАВИЛА ДЛЯ РЕКВИЗИТОВ:
1. Если блок отсутствует - верни null
2. Даты - СТРОГО "ДД.ММ.ГГГГ"
3. ИНН, КПП, БИК, счета - ТОЛЬКО цифры без пробелов
4. Названия организаций - ТОЧНО как написано с кавычками

СПЕЦИАЛЬНЫЕ СЛУЧАИ:
- Только таблица без реквизитов: "document_json": null
- Только реквизиты без таблицы: "table_items": null, "totals": null
- Рукописный текст ИГНОРИРУЙ

Верни ТОЛЬКО JSON без markdown разметки."""

def encode_img_base64(image_path: str) -> str:
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()


def run_qwen(image_path: str, api_key: str) -> str:
    config = load_config()["qwen"]
    img_base64 = encode_img_base64(image_path)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": config["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ],
            }
        ],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
    }

    response = requests.post(
        config["api_url"],
        headers=headers,
        json=payload,
        timeout=120,
    )

    if response.status_code != 200:
        raise Exception(f"Ошибка API {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"].strip()


def run_local(image_path: str) -> str:
    config = load_config()["local"]
    img_base64 = encode_img_base64(image_path)

    client = OpenAI(
        api_key="EMPTY",
        base_url=config["base_url"],
        timeout=3600,
    )

    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ],
            }
        ],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )

    return response.choices[0].message.content.strip()