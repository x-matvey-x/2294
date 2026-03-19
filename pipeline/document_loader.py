import os
import zipfile
from io import BytesIO

import fitz
from PIL import Image


def pdf_to_images(pdf_bytes: bytes, save_dir: str, filename: str) -> list[str]:
    doc_name = os.path.splitext(filename)[0]
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    saved_paths = []

    for i, page in enumerate(doc):
        mat = fitz.Matrix(3.0, 3.0)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_filename = f"{doc_name}_page{i + 1}.jpg"
        path = os.path.join(save_dir, page_filename)
        img.save(path, "JPEG", quality=95, optimize=True, subsampling=0)
        saved_paths.append(path)

    doc.close()
    return saved_paths


def zip_to_images(zip_bytes: bytes, save_dir: str, filename: str) -> list[str]:
    doc_name = os.path.splitext(filename)[0]
    saved_paths = []

    with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
        for i, file_info in enumerate(z.infolist()):
            if file_info.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                with z.open(file_info) as img_file:
                    img_data = img_file.read()
                img = Image.open(BytesIO(img_data))
                img.load()
                new_size = (img.width * 2, img.height * 2)
                img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
                page_filename = f"{doc_name}_page{i + 1}.jpg"
                new_path = os.path.join(save_dir, page_filename)
                img_resized.save(new_path, "JPEG", quality=95)
                saved_paths.append(new_path)

    return saved_paths


def single_image(image_bytes: bytes, filename: str, save_dir: str) -> list[str]:
    doc_name = os.path.splitext(filename)[0]
    page_filename = f"{doc_name}_page1.jpg"
    path = os.path.join(save_dir, page_filename)
    img = Image.open(BytesIO(image_bytes))
    new_size = (img.width * 2, img.height * 2)
    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
    img_resized.save(path, "JPEG", quality=95)
    return [path]


def load_document(file_bytes: bytes, filename: str, save_dir: str) -> list[str]:
    """
    Универсальная функция — сама определяет тип файла и возвращает список путей к картинкам.
    """
    os.makedirs(save_dir, exist_ok=True)
    filename_lower = filename.lower()

    if filename_lower.endswith(".pdf"):
        return pdf_to_images(file_bytes, save_dir, filename)
    elif filename_lower.endswith(".zip"):
        return zip_to_images(file_bytes, save_dir, filename)
    elif filename_lower.endswith((".png", ".jpg", ".jpeg")):
        return single_image(file_bytes, filename, save_dir)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {filename}")