import streamlit as st
import fitz
from PIL import Image
import os
import zipfile
from io import BytesIO


output_dir = "saved_images"
os.makedirs(output_dir, exist_ok=True)


def pdf_to_images(pdf_bytes, save_dir, filename):
    doc_name = os.path.splitext(filename)[0]
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    saved_paths = []
    for i, page in enumerate(doc):
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_filename = f"{doc_name}_page{i+1}.jpg"
        path = os.path.join(save_dir, page_filename)
        img.save(path, "JPEG", quality=95, optimize=True, subsampling=0)
        saved_paths.append(path)
    return saved_paths


def zip_to_images(zip_bytes, save_dir, filename):
    doc_name = os.path.splitext(filename)[0]
    saved_paths = []
    with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
        for i, file_info in enumerate(z.infolist()):
            if file_info.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                with z.open(file_info) as img_file:
                    img = Image.open(img_file)

                page_filename = f"{doc_name}_page{i+1}.jpg"
                new_path = os.path.join(save_dir, page_filename)
                img.save(new_path, "JPEG", quality=95, subsampling=0)
                saved_paths.append(new_path)
    return saved_paths


def single_image(image_bytes, filename, save_dir):
    doc_name = os.path.splitext(filename)[0]
    page_filename = f"{doc_name}_page1.jpg"
    path = os.path.join(save_dir, page_filename)
    img = Image.open(BytesIO(image_bytes))

    img.save(path, "JPEG", quality=95, optimize=True, subsampling=0)
    return [path]


def main():
    st.title("converter")

    uploaded_file = st.file_uploader("Загрузите файл", type=["pdf", "zip", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        filename = uploaded_file.name
        file_bytes = uploaded_file.read()

        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != filename:
            st.session_state.page = 0
            st.session_state.last_uploaded = filename

        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))

        filename_lower = filename.lower()
        if filename_lower.endswith(".pdf"):
            image_paths = pdf_to_images(file_bytes, output_dir, filename)
            st.write("Обработан PDF.")
        elif filename_lower.endswith(".zip"):
            image_paths = zip_to_images(file_bytes, output_dir, filename)
            st.write("Обработан ZIP архив с изображениями.")
        elif filename_lower.endswith((".png", ".jpg", ".jpeg")):
            image_paths = single_image(file_bytes, filename, output_dir)
            st.write("Обработана одиночная картинка.")
        else:
            st.error("Тип файла не поддерживается.")
            return

        if "page" not in st.session_state:
            st.session_state.page = 0

        if len(image_paths) > 1:
            st.markdown("""
            <style>
            .container {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 30px;
                margin-bottom: 20px;
            }
            .button-container {
                flex: 0 0 100px;
                display: flex;
                justify-content: center;
            }
            .image-container {
                flex: 1;
                display: flex;
                justify-content: center;
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown('<div class="container">', unsafe_allow_html=True)

            with st.container():
                left_col, image_col, right_col = st.columns([1, 6, 1])

                with left_col:
                    if st.button("Назад"):
                        st.session_state.page = (st.session_state.page - 1) % len(image_paths)

                with image_col:
                    st.image(
                        image_paths[st.session_state.page],
                        caption=f"Страница {st.session_state.page + 1} из {len(image_paths)} ({os.path.basename(image_paths[st.session_state.page])})",
                        use_container_width=True,
                    )

                with right_col:
                    if st.button("Вперёд"):
                        st.session_state.page = (st.session_state.page + 1) % len(image_paths)

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.image(
                image_paths[0],
                caption=os.path.basename(image_paths[0]),
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
