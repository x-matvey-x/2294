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


PROMPT = """You are a specialized in invoice and your role is to extract information from any invoice that is provided to you in the following valid json format. if the corresponding value is not present, leave the key with empty string.\n\n{"header": {"invoice_no": "номер счёта", "invoice_date": "дата выставления", "bank_name": "банк поставщика", "bik": "БИК банка", "corr_account": "корр. счёт", "recipient_account": "расчётный счёт получателя", "seller": "наименование поставщика", "seller_inn": "ИНН поставщика", "seller_kpp": "КПП поставщика", "seller_address": "адрес поставщика", "seller_phone": "телефон поставщика", "client": "наименование покупателя", "client_inn": "ИНН покупателя", "client_kpp": "КПП покупателя", "client_address": "адрес покупателя", "client_phone": "телефон покупателя", "basis": "основание платежа"}, "items": [{"item_position": "порядковый номер", "item_desc": "наименование товара/услуги", "item_qty": "количество", "item_unit": "единица измерения", "item_price": "цена за единицу", "item_amount": "сумма по строке", "item_vat_rate": "ставка НДС", "item_vat_amount": "сумма НДС", "item_total_with_vat": "сумма с НДС", "item_article": "артикул", "item_code": "код товара", "item_product_code": "код вида товара", "item_country_name": "страна происхождения", "item_country_code": "код страны", "item_customs_declaration": "таможенная декларация", "item_excise": "акциз"}], "summary": {"total_without_vat": "итого без НДС", "total_vat": "итого НДС", "total_vat_5": "НДС 5%", "total_vat_10": "НДС 10%", "total_vat_20": "НДС 20%", "total_vat_22": "НДС 22%", "total_with_vat": "итого с НДС"}}\n\nFill the keys only when the information is available.\n."""

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