import streamlit as st
import fitz
from PIL import Image
import os
import zipfile
from io import BytesIO, StringIO
import json
import pandas as pd
from dotenv import load_dotenv
from llm import qwen_ocr

load_dotenv()

st.set_page_config(layout="wide", page_title="OCR Converter")

output_dir = "saved_images"
os.makedirs(output_dir, exist_ok=True)


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


def main():
    st.title("OCR Converter")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.warning("OpenRouter API ключ не найден в .env файле")
        api_key = st.text_input("Введите OpenRouter API ключ:", type="password")
        if not api_key:
            st.stop()
    
    uploaded_file = st.file_uploader("Загрузите файл", type=["pdf", "zip", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        filename = uploaded_file.name
        file_bytes = uploaded_file.read()
        
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != filename:
            st.session_state.page = 0
            st.session_state.last_uploaded = filename
            st.session_state.ocr_results = {}
        
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
        
        if "page" not in st.session_state:
            st.session_state.page = 0

        if len(image_paths) > 1:
            col_prev, col_info, col_next = st.columns([1, 6, 1])
            with col_prev:
                if st.button("Назад"):
                    st.session_state.page = (st.session_state.page - 1) % len(image_paths)
                    st.rerun()
            with col_info:
                st.markdown(f"""<div style='text-align: center; background-color: transparent; padding: 10px; border-radius: 5px;'>
                    Страница {st.session_state.page + 1} из {len(image_paths)}</div>""", unsafe_allow_html=True)
            with col_next:
                if st.button("Вперёд"):
                    st.session_state.page = (st.session_state.page + 1) % len(image_paths)
                    st.rerun()

        col_image, col_results = st.columns([1, 1])
        
        with col_image:
            st.subheader("Документ")
            current_image_path = image_paths[st.session_state.page]
            img = Image.open(current_image_path)
            st.image(
                img,
                caption=os.path.basename(current_image_path),
                use_container_width=True
            )
        
        with col_results:
            #кнопка распознавания
            if st.session_state.page not in st.session_state.ocr_results:
                if st.button("Распознать текст", key=f"ocr_btn_{st.session_state.page}", use_container_width=True):
                    with st.spinner("Распознавание документа..."):
                        ocr_data = qwen_ocr(
                            current_image_path,
                            filename,
                            st.session_state.page,
                            len(image_paths),
                            api_key
                        )
                        st.session_state.ocr_results[st.session_state.page] = ocr_data
                        st.rerun()
            
            #отображение результатов
            if st.session_state.page in st.session_state.ocr_results:
                ocr_data = st.session_state.ocr_results[st.session_state.page]
                base_name = os.path.splitext(filename)[0]
                
                #проверка ошибок
                if ocr_data.get("metadata", {}).get("error"):
                    st.error(f"Ошибка: {ocr_data['metadata']['error']}")
                    if ocr_data["metadata"].get("raw_response"):
                        with st.expander("Показать сырой ответ модели"):
                            st.code(ocr_data["metadata"]["raw_response"])
                    
                    if st.button("Попробовать снова", key=f"retry_btn_{st.session_state.page}"):
                        del st.session_state.ocr_results[st.session_state.page]
                        st.rerun()
                    return
                
                #json 
                st.subheader("JSON")
                
                doc_json = ocr_data.get("document_json")
                if doc_json:
                    json_key = f"edited_json_{st.session_state.page}"
                    
                    if json_key not in st.session_state:
                        st.session_state[json_key] = json.dumps(doc_json, ensure_ascii=False, indent=2)
                    
                    edited_json = st.text_area(
                        "Редактировать JSON", 
                        value=st.session_state[json_key],
                        height=300,
                        key=f"json_editor_{st.session_state.page}"
                    )
                    
                    st.session_state[json_key] = edited_json
                    
                    st.download_button(
                        label="Скачать JSON",
                        data=st.session_state[json_key],
                        file_name=f"{base_name}_page{st.session_state.page + 1}.json",
                        mime="application/json",
                        key=f"download_json_{st.session_state.page}",
                        use_container_width=True
                    )
                else:
                    st.info("Информация о документе не найдена")
                
                st.divider()
                
                #csv
                st.subheader("CSV")
                
                table_data = ocr_data.get("table_csv")
                if table_data and len(table_data) > 0:
                    df_table = pd.DataFrame(table_data)
                    
                    st.caption(f"Столбцов: {len(df_table.columns)} | Строк: {len(df_table)}")
                    
                    edited_df = st.data_editor(
                        df_table,
                        num_rows="dynamic",
                        use_container_width=True,
                        key=f"csv_editor_{st.session_state.page}"
                    )
                    
                    # if ocr_data.get("totals"):
                    #     st.markdown("**Итоговые суммы:**")
                    #     totals_df = pd.DataFrame([ocr_data["totals"]])
                    #     st.dataframe(totals_df, use_container_width=True, hide_index=True)
                    
                    csv_buffer = StringIO()
                    edited_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                    csv_data = csv_buffer.getvalue().encode("utf-8-sig")
                    
                    st.download_button(
                        label="Скачать CSV",
                        data=csv_data,
                        file_name=f"{base_name}_page{st.session_state.page + 1}.csv",
                        mime="text/csv",
                        key=f"download_csv_{st.session_state.page}",
                        use_container_width=True
                    )
                else:
                    st.info("Таблица не найдена на документе")
            
            


if __name__ == "__main__":
    main()
