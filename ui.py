import streamlit as st
import requests
import base64
import pandas as pd
import os

# --- УМНАЯ НАСТРОЙКА URL ---
# Если переменная окружения API_URL задана (Docker), берем её.
# Если нет (Локально) - используем localhost.
API_URL = os.getenv("API_URL", "http://localhost:8000")
ENDPOINT = f"{API_URL}/analyze"

st.set_page_config(layout="wide", page_title="BPMN AI Analyzer")

st.title("🤖 BPMN AI Analyzer: Hybrid Vision Pipeline")
st.markdown(f"**Status:** Connecting to backend at `{API_URL}`")

uploaded_file = st.file_uploader("Загрузите изображение BPMN схемы", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Разделяем экран на две колонки
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Исходное изображение")
        st.image(uploaded_file, use_container_width=True)
        
        if st.button("🚀 Запустить анализ", type="primary"):
            with st.spinner("Processing... YOLOv11 finding nodes -> LSD finding lines -> Qwen assembling logic..."):
                try:
                    # Отправляем файл на бэкенд
                    files = {"file": uploaded_file.getvalue()}
                    response = requests.post(ENDPOINT, files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Отображаем результат (обработанную картинку)
                        img_data = base64.b64decode(data["image"])
                        st.subheader("👁️ Computer Vision (YOLO + LSD)")
                        st.image(img_data, caption="Green lines = Detected Connections", use_container_width=True)
                        
                        # Сохраняем логику в session state, чтобы не пропала
                        st.session_state['logic'] = data['logic']
                        st.session_state['raw'] = data['raw']
                        st.success("Analysis Complete!")
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}. Is the backend running?")

    with col2:
        st.subheader("🧠 Qwen 2.5-VL (Logics)")
        
        if 'logic' in st.session_state:
            logic_data = st.session_state['logic']
            
            if logic_data:
                # Красивая таблица
                df = pd.DataFrame(logic_data)
                st.dataframe(df, use_container_width=True)
                
                # Кнопка скачивания JSON
                json_str = pd.DataFrame(logic_data).to_json(orient="records", indent=2, force_ascii=False)
                st.download_button(
                    label="💾 Скачать JSON",
                    data=json_str,
                    file_name="bpmn_logic.json",
                    mime="application/json"
                )
            else:
                st.warning("Модель не нашла связей или вернула пустой список.")
                st.text_area("Raw Output", st.session_state.get('raw', ''), height=200)