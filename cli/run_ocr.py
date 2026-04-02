import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

from pipeline.pipeline import run

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="VLM OCR — распознавание документов")
    parser.add_argument("file", help="Путь к файлу (PDF, ZIP, PNG, JPG)")
    parser.add_argument("--model", choices=["local", "qwen"], default="local", help="Модель: local или qwen (default: local)")
    parser.add_argument("--output-dir", default="outputs", help="Папка для сохранения результатов (default: outputs)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.file):
        print(f"Ошибка: файл '{args.file}' не найден")
        sys.exit(1)

    use_local = args.model == "local"
    api_key = os.getenv("OPENROUTER_API_KEY") if not use_local else None

    if not use_local and not api_key:
        print("Ошибка: OPENROUTER_API_KEY не задан в .env")
        sys.exit(1)

    filename = os.path.basename(args.file)
    with open(args.file, "rb") as f:
        file_bytes = f.read()

    print(f"Файл: {filename}")
    print(f"Модель: {args.model}")
    print(f"Результаты: {args.output_dir}/")
    print("—" * 40)

    results = run(
        file_bytes=file_bytes,
        filename=filename,
        save_dir="saved_images",
        output_dir=args.output_dir,
        use_local=use_local,
        api_key=api_key,
    )

    print("—" * 40)
    for result in results:
        meta = result.get("metadata", {})
        page = meta.get("page", "?")
        total = meta.get("total_pages", "?")

        if meta.get("error"):
            print(f"Страница {page}/{total}: ошибка — {meta['error']}")
        else:
            exported = result.get("exported", {})
            json_path = exported.get("json_path")
            csv_path = exported.get("csv_path")
            parts = []
            if json_path:
                parts.append(f"JSON → {json_path}")
            if csv_path:
                parts.append(f"CSV → {csv_path}")
            if parts:
                print(f"Страница {page}/{total}: {', '.join(parts)}")
            else:
                print(f"Страница {page}/{total}: данные не найдены")

    print("—" * 40)
    print("Готово")


if __name__ == "__main__":
    main()