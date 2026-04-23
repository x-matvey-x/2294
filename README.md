# VLM OCR
 
Сервис для распознавания документов (счета, накладные, акты) на основе Vision Language Model. Извлекает реквизиты в JSON и таблицы в CSV.
 
## Структура проекта

```
vlm_ocr/
    pipeline/
        document_loader.py   # загрузка файлов, конвертация в картинки
        vlm_inference.py     # обращение к моделям (локальная / Qwen)
        postprocessing.py    # парсинг ответа модели
        export.py            # сохранение результатов в JSON
        pipeline.py          # функции run() и run_batch()

    api/
        server.py            # FastAPI сервер

    ui/
        streamlit_app.py     # веб-интерфейс

    cli/
        run_ocr.py           # CLI скрипт

    configs/
        model_config.yaml    # параметры моделей

    Dockerfile
    docker-compose.yaml
    requirements.txt
    .env
```

## Поддерживаемые форматы

- PDF
- PNG, JPG, JPEG
- ZIP с PDF документами (батч обработка)

## Требования

- Docker + Docker Compose
- Или Python 3.11+

## Старт (через Docker)

1. Создать `.env` файл в корне проекта:
```
OPENROUTER_API_KEY=your_key
```

2. Запустить:
```bash
docker-compose up --build
```

3. Открыть в браузере:
- Веб-интерфейс: http://localhost:8501
- API документация: http://localhost:8000/docs

## Модели

**Локальная (HunyuanOCR)** — работает через локальный сервер, адрес задаётся в `configs/model_config.yaml`. Не требует API ключа.

**Qwen (qwen-2.5-vl-72b-instruct)** — работает через OpenRouter. Требует `OPENROUTER_API_KEY` в `.env`.

## Веб-интерфейс

Две вкладки:

**Один документ** — загрузи PDF или картинку, нажми "Распознать текст". Результат появится постранично — реквизиты в JSON, позиции таблицы в CSV.

**Батч обработка** — загрузи ZIP архив с несколькими PDF документами. Каждый документ обработается по очереди, результаты сгруппированы по документам.

## API

### Проверка статуса
```
GET /health
```

### Распознавание одного документа
```
POST /ocr
```

Параметры:
- `file` — файл (PDF / PNG / JPG)
- `use_local` — `true` для локальной модели, `false` для Qwen (по умолчанию `true`)

Пример:
```bash
curl -X POST http://localhost:8000/ocr \
  -F "file=@invoice.pdf"
```

### Батч обработка
```
POST /ocr/batch
```

Параметры:
- `file` — ZIP архив с PDF документами внутри
- `use_local` — `true` для локальной модели, `false` для Qwen (по умолчанию `true`)

Пример:
```bash
curl -X POST http://localhost:8000/ocr/batch \
  -F "file=@documents.zip"
```

### Формат ответа
 
```json
{
  "filename": "invoice.pdf",
  "results": [
    {
      "document_json": {
        "Номер_счета": "1966/дА-23",
        "Дата_выставления": "07.08.2023",
        "Реквизиты": {},
        "Исполнитель": {},
        "Заказчик": {}
      },
      "table_items": [
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
      ],
      "totals": {
        "Итого": "192 000,00",
        "НДС_20": "32 000,00",
        "Всего_к_оплате": "192 000,00"
      },
      "metadata": {
        "document_name": "invoice",
        "page": 1,
        "total_pages": 3
      }
    }
  ]
}
```

## CLI

```bash
# один файл, локальная модель
python cli/run_ocr.py invoice.pdf

# один файл, Qwen
python cli/run_ocr.py invoice.pdf --model qwen

# батч — ZIP с PDF документами
python cli/run_ocr.py documents.zip --batch

# батч с указанием папки для результатов
python cli/run_ocr.py documents.zip --batch --output-dir my_results
```
