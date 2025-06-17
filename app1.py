import streamlit as st
import joblib
import tempfile
import os
import pdfplumber
import docx2txt
import pandas as pd
import base64

# Function to set background image using base64 encoding
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: white;
        text-shadow: 1px 1px 2px black;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set the background image
set_background("background.jpg")  # Change this to your image file name

# Load pipeline and label encoder
pipeline = joblib.load("cleantext_tfidf_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Title and description
st.title("📄 Resume Classifier")
st.write("Upload resume(s) in PDF, DOC, or DOCX format and get the predicted category.")

# Upload mode selection
upload_mode = st.radio("Choose upload mode:", ["Single Resume", "Multiple Resumes"])

# File upload
if upload_mode == "Single Resume":
    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "doc", "docx"], accept_multiple_files=False)
    uploaded_files = [uploaded_file] if uploaded_file else []
else:
    uploaded_files = st.file_uploader("Upload Multiple Resumes", type=["pdf", "doc", "docx"], accept_multiple_files=True)

# Resume text extractor
def extract_text(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    text = ""
    try:
        if file_type == "pdf":
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        elif file_type in ["doc", "docx"]:
            text = docx2txt.process(tmp_path)
        else:
            st.error("Unsupported file format!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
    finally:
        os.remove(tmp_path)
        
    return text.strip()

# Process uploaded files
if uploaded_files:
    results = []

    for uploaded_file in uploaded_files:
        if uploaded_file is None:
            continue

        resume_text = extract_text(uploaded_file)

        if resume_text == "":
            results.append({
                "File Name": uploaded_file.name,
                "Predicted Category": "No text found",
                "Confidence": "-"
            })
        else:
            prediction_proba = pipeline.predict_proba([resume_text])[0]
            max_prob = max(prediction_proba)
            predicted_index = prediction_proba.argmax()
            predicted_label = label_encoder.inverse_transform([predicted_index])[0]

            threshold = 0.8  # Confidence threshold

            if max_prob < threshold:
                predicted_label = "⚠️ Not confidently matched"
                confidence_display = f"{max_prob:.2f}"
            else:
                confidence_display = f"{max_prob:.2f}"

            results.append({
                "File Name": uploaded_file.name,
                "Predicted Category": predicted_label,
                "Confidence": confidence_display
            })

    if upload_mode == "Single Resume":
        result = results[0]
        st.subheader("🔍 Prediction Result")
        if result["Predicted Category"] == "No text found":
            st.warning("⚠️ No text found in the uploaded resume.")
        elif "Not confidently matched" in result["Predicted Category"]:
            st.warning("⚠️ The resume does not confidently match any known category.")
            st.write(f"**Confidence:** {result['Confidence']}")
        else:
            st.success(f"**Predicted Category:** {result['Predicted Category']}")
            st.write(f"**Confidence:** {result['Confidence']}")
    else:
        st.subheader("📊 Prediction Results Table")
        result_df = pd.DataFrame(results)
        st.dataframe(result_df, use_container_width=True)

        # Download button
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name="resume_predictions.csv",
            mime="text/csv"
        )

# About section
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
This Resume Classifier was created by:
- **Sanket Kshirsagar**
- **Abhijit Lavhale**
- **Anshul Sahare**
- **Harsha Chetlapalli**

Upload resumes to automatically classify them using a machine learning model trained on real-world resume data.
""")
