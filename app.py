import streamlit as st
import tensorflow as tf
import numpy as np
from keras_preprocessing import image
from PIL import Image
import os
from streamlit_option_menu import option_menu
import gdown
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

file_id2 = "1AjI-zbp-dBcFIIilD6u_z8HRm0oUYJ96"
output2 = "cat_dog_classifier.h5"  # เปลี่ยนเป็นไฟล์ .h5

# ดาวน์โหลดไฟล์จาก Google Drive
gdown.download(f"https://drive.google.com/uc?id={file_id2}", output2, quiet=False, verify=False)

file_id = "1ywoNNML2DF_kpROvOWSAZGH3-tJAsRX-"
output3 = "spam.csv"  # เปลี่ยนเป็นไฟล์ .h5

# ดาวน์โหลดไฟล์จาก Google Drive
gdown.download(f"https://drive.google.com/uc?id={file_id}", output3, quiet=False, verify=False)

# โหลดโมเดล Keras จากไฟล์ .h5


# โหลดโมเดลเพียงครั้งเดียว

model = tf.keras.models.load_model(output2)

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
if selected == "Neural Network":
    st.title("📸 อัปโหลดรูปภาพ")
    uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption="อัปโหลดรูปของคุณ", use_container_width = true)

        st.write("🔄 กำลังประมวลผล...")
        prediction = predict(image_display)

        if prediction > 0.5:
            st.success(f'🦴 นี่คือสุนัข! (ความมั่นใจ: {round(prediction * 100, 2)}%)')
        else:
            st.success(f'🐱 นี่คือแมว! (ความมั่นใจ: {round((1 - prediction) * 100, 2)}%)')

# หน้าเกี่ยวกับแอป
if selected == "Machine Learning":
    st.title("ℹ️ เกี่ยวกับแอปนี้")
    data = pd.read_csv(output3)
    data.drop_duplicates(inplace = True)
    data['Category'] = data['Category'].replace(['ham','spam'],['Not Spam','Spam'])
    # print(data.head())
    mess = data['Message']
    cat = data['Category']

    (mess_train, mess_test, cat_train, cat_test) = train_test_split(mess, cat, test_size=0.2)

    cv = CountVectorizer(stop_words='english')
    features= cv.fit_transform(mess_train)


    model = MultinomialNB()
    model.fit(features, cat_train)

    features_test = cv.transform(mess_test)
    # print(model.score(features_test, cat_test))
    def predict(message):
        input_message = cv.transform([message]).toarray()
        result = model.predict(input_message)
        return result

    st.header('Spam Detection')
    input_mess = st.text_input('Enter your message')

    if st.button('Validate'):
        output = predict(input_mess)
        st.markdown(output)

