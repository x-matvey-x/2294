import streamlit as st
import fitz
from PIL import Image
import os
import zipfile
from io import BytesIO, StringIO
import json
import pandas as pd
import time
import threading
from llm import qwen_ocr
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide", page_title="OCR Converter")

output_dir = "saved_images"
os.makedirs(output_dir, exist_ok=True)

def initialize_session_state():
    if "ocr_results" not in st.session_state:
        st.session_state.ocr_results = {}
    if "ocr_status" not in st.session_state:
        st.session_state.ocr_status = {} 
    if "ocr_progress" not in st.session_state:
        st.session_state.ocr_progress = {}
    if "edited_json" not in st.session_state:
        st.session_state.edited_json = {}
    if "edited_csv" not in st.session_state:
        st.session_state.edited_csv = {}
    if "image_paths" not in st.session_state:
        st.session_state.image_paths = []
    if "total_pages" not in st.session_state:
        st.session_state.total_pages = 0

def pdf_to_images(pdf_bytes, save_dir, filename):
    doc_name = os.path.splitext(filename)[0]
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    saved_paths = []
    for i, page in enumerate(doc):
        mat = fitz.Matrix(3.0, 3.0)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_filename = f"{doc_name}_page{i+1}.jpg"
        path = os.path.join(save_dir, page_filename)
        img.save(path, "JPEG", quality=95, optimize=True, subsampling=0)
        saved_paths.append(path)
    doc.close()
    return saved_paths

def zip_to_images(zip_bytes, save_dir, filename):
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
                page_filename = f"{doc_name}_page{i+1}.jpg"
                new_path = os.path.join(save_dir, page_filename)
                img_resized.save(new_path, "JPEG", quality=95)
                saved_paths.append(new_path)
    return saved_paths

def single_image(image_bytes, filename, save_dir):
    doc_name = os.path.splitext(filename)[0]
    page_filename = f"{doc_name}_page1.jpg"
    path = os.path.join(save_dir, page_filename)
    img = Image.open(BytesIO(image_bytes))
    new_size = (img.width * 2, img.height * 2)
    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
    img_resized.save(path, "JPEG", quality=95)
    return [path]

def ocr_worker(page_num, image_path, filename, total_pages, api_key, result_dict, status_dict, progress_dict):
    try:
        status_dict[page_num] = "processing"
        progress_dict[page_num] = 0

        for i in range(10):
            time.sleep(0.5)
            progress_dict[page_num] = (i + 1) / 10 * 100
        
        ocr_data = qwen_ocr(
            image_path,
            filename,
            page_num,
            total_pages,
            api_key
        )
        result_dict[page_num] = ocr_data
        status_dict[page_num] = "completed"
    except Exception as e:
        result_dict[page_num] = {"metadata": {"error": str(e)}}
        status_dict[page_num] = "error"

def start_ocr_background(page_num, image_path, filename, total_pages, api_key):
    if page_num not in st.session_state.ocr_status or st.session_state.ocr_status[page_num] in ["pending", "error"]:
        st.session_state.ocr_status[page_num] = "pending"
        st.session_state.ocr_progress[page_num] = 0

        result_dict = {}
        status_dict = {}
        progress_dict = {}
        
        thread = threading.Thread(
            target=ocr_worker,
            args=(page_num, image_path, filename, total_pages, api_key, result_dict, status_dict, progress_dict)
        )
        thread.daemon = True
        thread.start()
        
        st.session_state._ocr_result_dicts[page_num] = result_dict
        st.session_state._ocr_status_dicts[page_num] = status_dict
        st.session_state._ocr_progress_dicts[page_num] = progress_dict

def get_ocr_status(page_num):
    if page_num not in st.session_state.ocr_status:
        return "pending"
    
    status = st.session_state.ocr_status[page_num]
    
    if page_num in st.session_state._ocr_status_dicts:
        status = st.session_state._ocr_status_dicts[page_num].get(page_num, status)
    
    return status

def get_ocr_progress(page_num):
    if page_num in st.session_state._ocr_progress_dicts:
        return st.session_state._ocr_progress_dicts[page_num].get(page_num, 0)
    return st.session_state.ocr_progress.get(page_num, 0)

def main():
    st.title("OCR Converter")

    initialize_session_state()
    if "_ocr_result_dicts" not in st.session_state:
        st.session_state._ocr_result_dicts = {}
    if "_ocr_status_dicts" not in st.session_state:
        st.session_state._ocr_status_dicts = {}
    if "_ocr_progress_dicts" not in st.session_state:
        st.session_state._ocr_progress_dicts = {}

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["OPENROUTER_API_KEY"]
        except:
            pass

    uploaded_file = st.file_uploader("Загрузите файл", type=["pdf", "zip", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        filename = uploaded_file.name
        file_bytes = uploaded_file.read()

        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != filename:
            st.session_state.page = 0
            st.session_state.last_uploaded = filename
            st.session_state.ocr_results = {}
            st.session_state.ocr_status = {}
            st.session_state.ocr_progress = {}
            st.session_state.edited_json = {}
            st.session_state.edited_csv = {}
            st.session_state.image_paths = []
            st.session_state.total_pages = 0
            st.session_state._ocr_result_dicts = {}
            st.session_state._ocr_status_dicts = {}
            st.session_state._ocr_progress_dicts = {}

        if not st.session_state.image_paths:
            for f in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, f))

            filename_lower = filename.lower()
            if filename_lower.endswith(".pdf"):
                image_paths = pdf_to_images(file_bytes, output_dir, filename)
            elif filename_lower.endswith(".zip"):
                image_paths = zip_to_images(file_bytes, output_dir, filename)
            elif filename_lower.endswith((".png", ".jpg", ".jpeg")):
                image_paths = single_image(file_bytes, filename, output_dir)
            else:
                st.error("Тип файла не поддерживается.")
                return

            st.session_state.image_paths = image_paths
            st.session_state.total_pages = len(image_paths)

        image_paths = st.session_state.image_paths
        current_page = st.session_state.page

        if st.session_state.total_pages > 1:
            col_prev, col_info, col_next = st.columns([1, 6, 1])
            with col_prev:
                if st.button("Назад"):
                    st.session_state.page = (st.session_state.page - 1) % st.session_state.total_pages
                    st.rerun()
            with col_info:
                st.markdown(f"""<div style='text-align: center; background-color: transparent; padding: 10px; border-radius: 5px;'>
                    Страница {current_page + 1} из {st.session_state.total_pages}</div>""", unsafe_allow_html=True)
            with col_next:
                if st.button("Вперёд"):
                    st.session_state.page = (st.session_state.page + 1) % st.session_state.total_pages
                    st.rerun()

        col_image, col_results = st.columns([1, 1])

        with col_image:
            st.subheader("Документ")
            current_image_path = image_paths[current_page]
            img = Image.open(current_image_path)
            st.image(img, caption=os.path.basename(current_image_path), use_container_width=True)

        with col_results:
            base_name = os.path.splitext(filename)[0]
            status = get_ocr_status(current_page)

            if status == "pending":
                if st.button("🚀 Распознать текст", key=f"ocr_btn_{current_page}", use_container_width=True):
                    start_ocr_background(current_page, current_image_path, filename, st.session_state.total_pages, api_key)
                    st.rerun()
            
            if status == "processing":
                st_autorefresh(interval=1000, key=f"refresh_{current_page}")

                progress = get_ocr_progress(current_page)
                st.progress(progress / 100, text=f"Обработка... {progress:.0f}%")

            if status in ["completed", "error"]:
                ocr_data = st.session_state.ocr_results.get(current_page)
                if not ocr_data and current_page in st.session_state._ocr_result_dicts:
                    ocr_data = st.session_state._ocr_result_dicts[current_page].get(current_page)
                
                if ocr_data:

                    if ocr_data.get("metadata", {}).get("error"):
                        st.error(f"Ошибка: {ocr_data['metadata']['error']}")
                        if ocr_data["metadata"].get("raw_response"):
                            with st.expander("Показать сырой ответ модели"):
                                st.code(ocr_data["metadata"]["raw_response"])
                        
                        if st.button("🔄 Попробовать снова", key=f"retry_btn_{current_page}"):
                            st.session_state.ocr_status.pop(current_page, None)
                            st.session_state.ocr_results.pop(current_page, None)
                            st.rerun()
                        st.stop()
                    
                    st.subheader("JSON")
                    doc_json = ocr_data.get("document_json")
                    if doc_json:
                        json_key = f"edited_json_{current_page}"
                        if json_key not in st.session_state:
                            st.session_state[json_key] = json.dumps(doc_json, ensure_ascii=False, indent=2)
                        
                        edited_json = st.text_area(
                            "Редактировать JSON", 
                            value=st.session_state[json_key],
                            height=300,
                            key=f"json_editor_{current_page}"
                        )
                        st.session_state[json_key] = edited_json
                        
                        st.download_button(
                            label="Скачать JSON",
                            data=st.session_state[json_key],
                            file_name=f"{base_name}_page{current_page + 1}.json",
                            mime="application/json",
                            key=f"download_json_{current_page}",
                            use_container_width=True
                        )
                    else:
                        st.info("Информация не найдена на документе")
                    
                    st.divider()

                    st.subheader("CSV")
                    table_data = ocr_data.get("table_csv")
                    if table_data and len(table_data) > 0:
                        df_table = pd.DataFrame(table_data)
                        st.caption(f"Столбцов: {len(df_table.columns)} | Строк: {len(df_table)}")
                        
                        edited_df = st.data_editor(
                            df_table,
                            num_rows="dynamic",
                            use_container_width=True,
                            key=f"csv_editor_{current_page}"
                        )
                        
                        csv_buffer = StringIO()
                        edited_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        csv_data = csv_buffer.getvalue().encode("utf-8-sig")
                        
                        st.download_button(
                            label="Скачать CSV",
                            data=csv_data,
                            file_name=f"{base_name}_page{current_page + 1}.csv",
                            mime="text/csv",
                            key=f"download_csv_{current_page}",
                            use_container_width=True
                        )
                    else:
                        st.info("Таблица не найдена на документе")
                else:
                    st.warning("Данные не найдены. Нажмите 'Распознать текст'")


if __name__ == "__main__":
    main()
