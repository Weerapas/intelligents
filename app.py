import streamlit as st
import tensorflow as tf
import numpy as np
from keras_preprocessing import image
from PIL import Image
import os
from streamlit_option_menu import option_menu

# โหลดโมเดลเพียงครั้งเดียว
MODEL_PATH = 'cat_dog_classifier.h5'
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

def predict(img):
    img = img.resize((128, 128))  # ปรับขนาดให้ตรงกับโมเดล
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0  # ปรับ scale

    prediction = model.predict(img_array)
    return prediction.item()

# ส่วนของ Navigation Menu (Sidebar)
with st.sidebar:
    selected = option_menu(
        menu_title="เมนูหลัก",  
        options=["🏠 หน้าแรก", "Neural Network", "Machine Learning"],  
        icons=["house"],  
        menu_icon="cast",  
        default_index=0,  
    )

# หน้าแรก
if selected == "🏠 หน้าแรก":
    st.title("🐶🐱 Cat vs Dog Classifier")
    st.write("โมเดล AI สำหรับทำนายว่าเป็นแมวหรือสุนัข!")

# หน้าอัปโหลดและทำนายภาพ
elif selected == "Neural Network":
    st.title("📸 อัปโหลดรูปภาพ")
    uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption="อัปโหลดรูปของคุณ", use_column_width=True)

        st.write("🔄 กำลังประมวลผล...")
        prediction = predict(image_display)

        if prediction > 0.5:
            st.success(f'🦴 นี่คือสุนัข! (ความมั่นใจ: {round(prediction * 100, 2)}%)')
        else:
            st.success(f'🐱 นี่คือแมว! (ความมั่นใจ: {round((1 - prediction) * 100, 2)}%)')

# หน้าเกี่ยวกับแอป
elif selected == "Machine Learning":
    st.title("ℹ️ เกี่ยวกับแอปนี้")
    st.write("📌 แอปนี้ใช้โมเดล Deep Learning (CNN) ในการจำแนกภาพว่าเป็นแมวหรือสุนัข")
    st.write("👨‍💻 พัฒนาโดย [ใส่ชื่อคุณ]")

