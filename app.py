import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO

from image_processor import canny_sketch_pipeline

# streamlit

st.set_page_config(layout="wide")
st.title("Canny Sketch")

col1, col2 = st.columns(2)

with col1:
    st.header("Tải ảnh")
    uploaded_file = st.file_uploader("Tải ảnh lên (.jpg, .png, .jpeg)", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        # Đọc ảnh bằng PIL/NumPy
        img_pil = Image.open(uploaded_file).convert('RGB')
        img_array_rgb = np.array(img_pil)

        # SỬA LỖI 1: Thay use_column_width bằng use_container_width
        st.image(img_array_rgb, caption="Ảnh gốc", use_container_width=True)

        st.markdown("---")
        st.markdown("### 2. Tham số Canny")

        # Tùy chỉnh Gaussian Blur (sigma)
        sigma_gauss = st.slider("Sigma (Độ mờ Gaussian)", 0.5, 5.0, 1.4, 0.1)

        # Tùy chỉnh Double Threshold (Tỷ lệ)
        high_ratio = st.slider("Tỷ lệ Ngưỡng Cao (High Ratio)", 0.01, 0.5, 0.09)
        low_ratio = st.slider("Tỷ lệ Ngưỡng Thấp (Low Ratio)", 0.01, 0.2, 0.05)

        if st.button("Tạo Tranh Vẽ"):
            with st.spinner("Đang chạy Canny 6 bước..."):
                # GỌI HÀM XỬ LÝ ẢNH TỪ FILE image_processor.py
                result_sketch = canny_sketch_pipeline(
                    img_array_rgb,
                    sigma=sigma_gauss,
                    low_ratio=low_ratio,
                    high_ratio=high_ratio
                )
                st.session_state['result'] = result_sketch
                st.session_state['processed'] = True
                st.rerun()
    # else:
    #     st.warning("Vui lòng tải ảnh lên.")

# hiển thị kết quả
with col2:
    st.header("Kết quả Vẽ Tay")

    if 'processed' in st.session_state and st.session_state['processed']:
        result_sketch = st.session_state['result']


        st.image(result_sketch, caption="Kết quả Sketch Canny", use_container_width=True)

        # Tạo nút Tải xuống
        img_to_save = Image.fromarray(result_sketch)
        buf = BytesIO()
        img_to_save.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Tải ảnh về",
            data=byte_im,
            file_name="canny_scipy_sketch.png",
            mime="image/png"
        )
    else:
        st.info("Kết quả Canny Sketch sẽ hiển thị ở đây.")