import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import threading
import uuid
from io import StringIO

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_autorefresh import st_autorefresh

from pipeline.pipeline import run_page, run_batch
from pipeline.document_loader import load_document

st.set_page_config(layout="wide", page_title="OCR Converter")


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
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "local_ocr"
    if "session_dir" not in st.session_state:
        session_id = str(uuid.uuid4())
        st.session_state.session_dir = os.path.join("saved_images", session_id)
        os.makedirs(st.session_state.session_dir, exist_ok=True)
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = {}
    if "batch_status" not in st.session_state:
        st.session_state.batch_status = "idle"


# ─── Таблица реквизитов ──────────────────────────────────────────────────────

def render_requisites_table(doc_json: dict):
    requisites = doc_json.get("Реквизиты")
    if requisites and isinstance(requisites, dict):
        rows = [{"Поле": k, "Значение": str(v) if v is not None else ""} for k, v in requisites.items()]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Реквизиты не найдены")


# ─── OCR воркер ─────────────────────────────────────────────────────────────

def ocr_worker(page_num, image_path, filename, total_pages, api_key, result_dict, status_dict, model):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            status_dict[page_num] = "processing"
            use_local = model == "local_ocr"
            ocr_data = run_page(
                image_path=image_path,
                filename=filename,
                page_num=page_num,
                total_pages=total_pages,
                use_local=use_local,
                api_key=api_key,
            )

            if "document_json" in ocr_data and ocr_data["document_json"]:
                try:
                    json_str = json.dumps(ocr_data["document_json"])
                    json.loads(json_str)
                    result_dict[page_num] = ocr_data
                    status_dict[page_num] = "completed"
                    return
                except json.JSONDecodeError:
                    if attempt < max_retries - 1:
                        continue
                    else:
                        result_dict[page_num] = {
                            "metadata": {"error": "Невалидный JSON после всех попыток"},
                            "document_json": ocr_data.get("document_json"),
                        }
                        status_dict[page_num] = "error"
                        return
            else:
                result_dict[page_num] = ocr_data
                status_dict[page_num] = "completed"
                return

        except Exception as e:
            if attempt < max_retries - 1:
                continue
            else:
                result_dict[page_num] = {"metadata": {"error": str(e)}}
                status_dict[page_num] = "error"


def start_ocr_background(page_num, image_path, filename, total_pages, api_key, model):
    if page_num not in st.session_state.ocr_status or st.session_state.ocr_status[page_num] in ["pending", "error"]:
        st.session_state.ocr_status[page_num] = "processing"
        st.session_state.ocr_progress[page_num] = 0

        result_dict = {}
        status_dict = {}

        thread = threading.Thread(
            target=ocr_worker,
            args=(page_num, image_path, filename, total_pages, api_key, result_dict, status_dict, model),
        )
        thread.daemon = True
        thread.start()

        st.session_state._ocr_result_dicts[page_num] = result_dict
        st.session_state._ocr_status_dicts[page_num] = status_dict


def get_ocr_status(page_num):
    if page_num not in st.session_state.ocr_status:
        return "pending"
    status = st.session_state.ocr_status[page_num]
    if page_num in st.session_state._ocr_status_dicts:
        status = st.session_state._ocr_status_dicts[page_num].get(page_num, status)
    return status


# ─── Батч воркер ────────────────────────────────────────────────────────────

def batch_worker(file_bytes, api_key, model, result_store, status_store):
    try:
        use_local = model == "local_ocr"
        results = run_batch(
            zip_bytes=file_bytes,
            save_dir="saved_images",
            output_dir=None,
            use_local=use_local,
            api_key=api_key,
        )
        result_store["results"] = results
        status_store["status"] = "completed"
    except Exception as e:
        result_store["error"] = str(e)
        status_store["status"] = "error"


# ─── Отображение результата страницы ────────────────────────────────────────

def render_page_result(ocr_data, base_name, page_num):
    if ocr_data.get("metadata", {}).get("error"):
        st.error(f"Ошибка: {ocr_data['metadata']['error']}")
        if ocr_data["metadata"].get("raw_response"):
            with st.expander("Показать сырой ответ модели"):
                st.code(ocr_data["metadata"]["raw_response"])
        return

    doc_json = ocr_data.get("document_json")

    if doc_json:
        # Таблица реквизитов
        st.subheader("Реквизиты")
        render_requisites_table(doc_json)

        st.divider()

        # JSON редактор
        st.subheader("JSON")
        json_key = f"edited_json_{base_name}_{page_num}"
        if json_key not in st.session_state:
            st.session_state[json_key] = json.dumps(doc_json, ensure_ascii=False, indent=2)
        edited_json = st.text_area(
            "Редактировать JSON",
            value=st.session_state[json_key],
            height=300,
            key=f"json_editor_{base_name}_{page_num}",
        )
        st.session_state[json_key] = edited_json
        st.download_button(
            label="Скачать JSON",
            data=st.session_state[json_key],
            file_name=f"{base_name}_page{page_num + 1}.json",
            mime="application/json",
            key=f"download_json_{base_name}_{page_num}",
            use_container_width=True,
        )
    else:
        st.info("Информация не найдена на документе")

    st.divider()

    # Позиции таблицы
    st.subheader("Позиции таблицы")
    table_items = ocr_data.get("table_items")
    if table_items and len(table_items) > 0:
        df = pd.DataFrame(table_items)
        st.caption(f"Позиций: {len(df)}")
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            key=f"items_editor_{base_name}_{page_num}",
        )
        csv_buffer = StringIO()
        edited_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
        st.download_button(
            label="Скачать CSV",
            data=csv_buffer.getvalue().encode("utf-8-sig"),
            file_name=f"{base_name}_page{page_num + 1}_items.csv",
            mime="text/csv",
            key=f"download_items_{base_name}_{page_num}",
            use_container_width=True,
        )
    else:
        st.info("Таблица не найдена на документе")


# ─── Вкладка: один документ ─────────────────────────────────────────────────

def tab_single(api_key, model):
    uploaded_file = st.file_uploader(
        "Загрузите файл",
        type=["pdf", "png", "jpg", "jpeg"],
        key="single_uploader",
    )

    if uploaded_file is None:
        return

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

    if not st.session_state.image_paths:
        save_dir = st.session_state.session_dir
        for f in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, f))
        st.session_state.image_paths = load_document(file_bytes, filename, save_dir)
        st.session_state.total_pages = len(st.session_state.image_paths)

    image_paths = st.session_state.image_paths
    current_page = st.session_state.page

    if st.session_state.total_pages > 1:
        col_prev, col_info, col_next = st.columns([1, 6, 1])
        with col_prev:
            if st.button("Назад"):
                st.session_state.page = (st.session_state.page - 1) % st.session_state.total_pages
                st.rerun()
        with col_info:
            st.markdown(
                f"<div style='text-align: center; padding: 10px;'>Страница {current_page + 1} из {st.session_state.total_pages}</div>",
                unsafe_allow_html=True,
            )
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
                start_ocr_background(
                    current_page, current_image_path, filename,
                    st.session_state.total_pages, api_key, model,
                )
                st.rerun()

        if status == "processing":
            st_autorefresh(interval=1000, key=f"refresh_{current_page}")
            if current_page in st.session_state._ocr_status_dicts:
                thread_status = st.session_state._ocr_status_dicts[current_page].get(current_page)
                if thread_status == "completed":
                    result = st.session_state._ocr_result_dicts[current_page].get(current_page)
                    st.session_state.ocr_results[current_page] = result
                    st.session_state.ocr_status[current_page] = "completed"
                    st.rerun()
                elif thread_status == "error":
                    result = st.session_state._ocr_result_dicts[current_page].get(current_page)
                    st.session_state.ocr_results[current_page] = result
                    st.session_state.ocr_status[current_page] = "error"
                    st.rerun()
            st.markdown("""
                <div style="text-align: center; padding: 20px;">
                    <div style="border: 4px solid #f3f3f3; border-top: 4px solid #3498db;
                                border-radius: 50%; width: 40px; height: 40px;
                                animation: spin 1s linear infinite; margin: 0 auto;"></div>
                    <p style="margin-top: 10px;">Распознавание текста...</p>
                </div>
                <style>@keyframes spin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}</style>
            """, unsafe_allow_html=True)

        if status in ["completed", "error"]:
            ocr_data = st.session_state.ocr_results.get(current_page)
            if not ocr_data and current_page in st.session_state._ocr_result_dicts:
                ocr_data = st.session_state._ocr_result_dicts[current_page].get(current_page)

            if ocr_data:
                if st.button("🔄 Распознать заново", key=f"retry_btn_{current_page}"):
                    st.session_state.ocr_status.pop(current_page, None)
                    st.session_state.ocr_results.pop(current_page, None)
                    st.rerun()
                render_page_result(ocr_data, base_name, current_page)
            else:
                st.warning("Данные не найдены. Нажмите 'Распознать текст'")


# ─── Вкладка: батч ──────────────────────────────────────────────────────────

def tab_batch(api_key, model):
    uploaded_zip = st.file_uploader(
        "Загрузите ZIP архив с PDF документами",
        type=["zip"],
        key="batch_uploader",
    )

    if uploaded_zip is None:
        return

    if "last_batch_file" not in st.session_state or st.session_state.last_batch_file != uploaded_zip.name:
        st.session_state.last_batch_file = uploaded_zip.name
        st.session_state.batch_results = {}
        st.session_state.batch_status = "idle"
        st.session_state._batch_result_store = {}
        st.session_state._batch_status_store = {}

    status = st.session_state.batch_status

    if status == "idle":
        if st.button("🚀 Запустить батч обработку", use_container_width=True):
            file_bytes = uploaded_zip.read()
            result_store = {}
            status_store = {}
            st.session_state._batch_result_store = result_store
            st.session_state._batch_status_store = status_store
            st.session_state.batch_status = "processing"

            thread = threading.Thread(
                target=batch_worker,
                args=(file_bytes, api_key, model, result_store, status_store),
            )
            thread.daemon = True
            thread.start()
            st.rerun()

    if status == "processing":
        st_autorefresh(interval=2000, key="batch_refresh")

        thread_status = st.session_state._batch_status_store.get("status")
        if thread_status == "completed":
            st.session_state.batch_results = st.session_state._batch_result_store.get("results", {})
            st.session_state.batch_status = "completed"
            st.rerun()
        elif thread_status == "error":
            st.session_state.batch_status = "error"
            st.rerun()

        st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <div style="border: 4px solid #f3f3f3; border-top: 4px solid #3498db;
                            border-radius: 50%; width: 40px; height: 40px;
                            animation: spin 1s linear infinite; margin: 0 auto;"></div>
                <p style="margin-top: 10px;">Обрабатываю документы...</p>
            </div>
            <style>@keyframes spin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}</style>
        """, unsafe_allow_html=True)

    if status == "error":
        error = st.session_state._batch_result_store.get("error", "Неизвестная ошибка")
        st.error(f"Ошибка: {error}")
        if st.button("🔄 Попробовать снова"):
            st.session_state.batch_status = "idle"
            st.rerun()

    if status == "completed":
        results = st.session_state.batch_results
        st.success(f"Обработано документов: {len(results)}")

        if st.button("🔄 Обработать заново"):
            st.session_state.batch_status = "idle"
            st.rerun()

        for doc_name, pages in results.items():
            with st.expander(f"📄 {doc_name} — страниц: {len(pages)}", expanded=False):
                for page_result in pages:
                    page_num = page_result.get("metadata", {}).get("page", 1) - 1
                    st.markdown(f"**Страница {page_num + 1}**")
                    base_name = os.path.splitext(doc_name)[0]
                    render_page_result(page_result, base_name, page_num)
                    st.divider()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    st.title("OCR Converter")

    initialize_session_state()
    if "_ocr_result_dicts" not in st.session_state:
        st.session_state._ocr_result_dicts = {}
    if "_ocr_status_dicts" not in st.session_state:
        st.session_state._ocr_status_dicts = {}
    if "_batch_result_store" not in st.session_state:
        st.session_state._batch_result_store = {}
    if "_batch_status_store" not in st.session_state:
        st.session_state._batch_status_store = {}

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["OPENROUTER_API_KEY"]
        except Exception:
            pass

    model_options = {"Local OCR": "local_ocr", "Qwen OCR": "qwen_ocr"}
    selected_label = st.selectbox(
        "Модель распознавания",
        options=list(model_options.keys()),
        index=0 if st.session_state.selected_model == "local_ocr" else 1,
    )
    st.session_state.selected_model = model_options[selected_label]
    model = st.session_state.selected_model

    tab1, tab2 = st.tabs(["Один документ", "Батч обработка"])

    with tab1:
        tab_single(api_key, model)

    with tab2:
        tab_batch(api_key, model)


if __name__ == "__main__":
    main()