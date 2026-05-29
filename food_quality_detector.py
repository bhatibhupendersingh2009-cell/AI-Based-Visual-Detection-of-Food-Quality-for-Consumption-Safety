import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Food Quality Detection",
    page_icon="🍎",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = load_model("food_quality_model.keras")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background: linear-gradient(to right, #eef4ff, #f8fbff);
}

.title{
    font-size:55px;
    font-weight:bold;
    color:#0B1F66;
    text-align:center;
}

.subtitle{
    text-align:center;
    font-size:22px;
    color:#444;
    margin-bottom:30px;
}

.card{
    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.1);
}

.result-fresh{
    background:linear-gradient(to right,#28a745,#5dd879);
    padding:25px;
    border-radius:20px;
    color:white;
    text-align:center;
    font-size:35px;
    font-weight:bold;
    box-shadow:0px 5px 20px rgba(0,0,0,0.2);
}

.result-spoiled{
    background:linear-gradient(to right,#dc3545,#ff6b6b);
    padding:25px;
    border-radius:20px;
    color:white;
    text-align:center;
    font-size:35px;
    font-weight:bold;
    box-shadow:0px 5px 20px rgba(0,0,0,0.2);
}

.feature-card{
    background:white;
    padding:20px;
    border-radius:18px;
    text-align:center;
    box-shadow:0px 4px 15px rgba(0,0,0,0.1);
    transition:0.3s;
}

.feature-card:hover{
    transform:scale(1.03);
}

.footer{
    text-align:center;
    color:gray;
    font-size:16px;
    margin-top:40px;
}

[data-testid="stFileUploader"]{
    padding:25px;
    border-radius:20px;
    border:3px dashed #4A90E2;
    background-color:white;
    box-shadow:0px 5px 15px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown(
    "<div class='title'>🍎 AI Food Quality Detection</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI-Based Visual Detection of Food Quality for Consumption Safety</div>",
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Project Details")

st.sidebar.info("""
### Technologies Used
- Python
- TensorFlow
- CNN / MobileNetV2
- Streamlit
- NumPy
""")

st.sidebar.success("Presented By Bhupender Singh")

# ---------------- MAIN SECTION ----------------
col1, col2 = st.columns([1,1])

with col1:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📂 Upload Food Image",
        type=["jpg","jpeg","png"],
        help="Upload Fresh or Spoiled Food Image"
    )

    if uploaded_file is not None:

        img = Image.open(uploaded_file)

        st.image(
            img,
            caption="Uploaded Food Image",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if uploaded_file is not None:

    # IMAGE PREPROCESSING
    img = img.resize((224,224))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    # LOADING EFFECT
    with st.spinner("🤖 AI Model is Analyzing Food Quality..."):
        time.sleep(2)

        prediction = model.predict(img_array)

    prediction_value = prediction[0][0]

    # RESULT LOGIC
    if prediction_value >= 0.5:

        result = "Spoiled Food ❌"

        confidence = prediction_value * 100

        result_class = "result-spoiled"

    else:

        result = "Fresh Food ✅"

        confidence = (1 - prediction_value) * 100

        result_class = "result-fresh"

    # RESULT DISPLAY
    with col2:

        st.markdown(
            f"""
            <div class='{result_class}'>
                {result}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.subheader("📊 Confidence Score")

        st.progress(int(confidence))

        st.success(
            f"Confidence : {round(confidence,2)}%"
        )

        st.metric(
            label="Prediction Accuracy",
            value=f"{round(confidence,2)}%"
        )

# ---------------- FEATURES ----------------
st.write("")
st.write("")

st.markdown("## 🚀 Key Features")

f1, f2, f3 = st.columns(3)

with f1:
    st.markdown("""
    <div class='feature-card'>
    <h2>📤</h2>
    <h3>Easy Upload</h3>
    <p>Upload food images instantly using a user-friendly interface.</p>
    </div>
    """, unsafe_allow_html=True)

with f2:
    st.markdown("""
    <div class='feature-card'>
    <h2>🤖</h2>
    <h3>AI Detection</h3>
    <p>Deep learning model predicts Fresh or Spoiled food accurately.</p>
    </div>
    """, unsafe_allow_html=True)

with f3:
    st.markdown("""
    <div class='feature-card'>
    <h2>⚡</h2>
    <h3>Fast Results</h3>
    <p>Get instant prediction results with confidence score.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- ABOUT PROJECT ----------------
st.write("")
st.write("")

st.markdown("## 📖 About Project")

st.markdown("""
<div class='card'>
This AI-based system helps in detecting food freshness using Deep Learning and Computer Vision.
Users can upload food images and the AI model predicts whether the food is Fresh or Spoiled with high accuracy.
</div>
""", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(
    "<div class='footer'>🍽️ AI-Based Visual Detection of Food Quality for Consumption Safety</div>",
    unsafe_allow_html=True
)