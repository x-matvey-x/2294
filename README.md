# VLM OCR
 
Сервис для распознавания документов (счета, накладные, акты) на основе Vision Language Model. Извлекает реквизиты в JSON и таблицы в CSV.
 
## Структура проекта
 
```
vlm_ocr/
    pipeline/
        document_loader.py   # загрузка файлов, конвертация в картинки
        vlm_inference.py     # обращение к моделям (локальная / Qwen)
        postprocessing.py    # парсинг ответа модели
        export.py            # сохранение результатов в JSON / CSV
        pipeline.py          # главная функция run(), связывает всё вместе
 
    api/
        server.py            # FastAPI сервер
 
    ui/
        streamlit_app.py     # веб-интерфейс
 
    configs/
        model_config.yaml    # параметры моделей
 
    Dockerfile
    docker-compose.yaml
    requirements.txt
    .env
```
 
## Поддерживаемые форматы
 
- PDF
- ZIP с изображениями
- PNG, JPG, JPEG
 
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
 
**Локальная (HunyuanOCR)** — работает через локальный сервер, адрес задаётся в `vlm_inference.py`. Не требует API ключа.
 
**Qwen (qwen-2.5-vl-72b-instruct)** — работает через OpenRouter. Требует `OPENROUTER_API_KEY` в `.env`.
 
## API
 
### Проверка статуса
```
GET /health
```
 
### Распознавание документа
```
POST /ocr
```
 
Параметры:
- `file` — файл (PDF / ZIP / PNG / JPG)
- `use_local` — `true` для локальной модели, `false` для Qwen (по умолчанию `true`)
 
Пример запроса:
```bash
curl -X POST http://localhost:8000/ocr \
  -F "file=@invoice.pdf"
```

### Формат ответа
 
```json
{
  "results": [
    {
      "document_json": {
        "Номер_счета": "1966/дА-23",
        "Дата_выставления": "07.08.2023",
        "Реквизиты": { ... },
        "Исполнитель": { ... },
        "Заказчик": { ... }
      },
      "table_csv": [
        { "№": 1, "Товары (работы, услуги)": "...", "Цена": 1000.00 }
      ],
      "totals": {
        "Итого": 192000.00,
        "НДС_20": 32000.00,
        "Всего_к_оплате": 192000.00
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
 
## Использование pipeline напрямую
 
```python
from pipeline.pipeline import run
 
with open("invoice.pdf", "rb") as f:
    file_bytes = f.read()
 
results = run(
    file_bytes=file_bytes,
    filename="invoice.pdf",
    save_dir="saved_images",
    output_dir="results",
    use_local=True,
    api_key=None,
)
```