import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from dotenv import load_dotenv
from pipeline.pipeline import run, run_batch

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="VLM OCR — распознавание документов")
    parser.add_argument("file", help="Путь к файлу (PDF, PNG, JPG) или ZIP архиву для батч обработки")
    parser.add_argument("--model", choices=["local", "qwen"], default="local", help="Модель: local или qwen (default: local)")
    parser.add_argument("--output-dir", default="outputs", help="Папка для сохранения результатов (default: outputs)")
    parser.add_argument("--batch", action="store_true", help="Батч режим — ZIP архив с PDF документами внутри")
    return parser.parse_args()


def get_api_key(use_local):
    if use_local:
        return None
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Ошибка: OPENROUTER_API_KEY не задан в .env")
        sys.exit(1)
    return api_key


def print_page_result(result, doc_name=None):
    meta = result.get("metadata", {})
    page = meta.get("page", "?")
    total = meta.get("total_pages", "?")
    label = f"[{doc_name}] " if doc_name else ""

    if meta.get("error"):
        print(f"  {label}Страница {page}/{total}: ошибка — {meta['error']}")
        return

    exported = result.get("exported", {})
    json_path = exported.get("json_path")
    items_path = exported.get("items_path")
    parts = []
    if json_path:
        parts.append(f"JSON → {json_path}")
    if items_path:
        parts.append(f"items → {items_path}")
    if parts:
        print(f"  {label}Страница {page}/{total}: {', '.join(parts)}")
    else:
        print(f"  {label}Страница {page}/{total}: данные не найдены")


def run_single(args, use_local, api_key):
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
        print_page_result(result)


def run_batch_mode(args, use_local, api_key):
    with open(args.file, "rb") as f:
        zip_bytes = f.read()

    print(f"Архив: {args.file}")
    print(f"Модель: {args.model}")
    print(f"Результаты: {args.output_dir}/")
    print("—" * 40)

    results = run_batch(
        zip_bytes=zip_bytes,
        save_dir="saved_images",
        output_dir=args.output_dir,
        use_local=use_local,
        api_key=api_key,
    )

    print("—" * 40)
    for doc_name, pages in results.items():
        print(f"📄 {doc_name}:")
        for result in pages:
            print_page_result(result, doc_name=None)

    print(f"\nОбработано документов: {len(results)}")


def main():
    args = parse_args()

    if not os.path.exists(args.file):
        print(f"Ошибка: файл '{args.file}' не найден")
        sys.exit(1)

    if args.batch and not args.file.lower().endswith(".zip"):
        print("Ошибка: батч режим работает только с ZIP архивами")
        sys.exit(1)

    use_local = args.model == "local"
    api_key = get_api_key(use_local)

    os.makedirs("saved_images", exist_ok=True)

    if args.batch:
        run_batch_mode(args, use_local, api_key)
    else:
        run_single(args, use_local, api_key)

    print("—" * 40)
    print("Готово")


if __name__ == "__main__":
    main()