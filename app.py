import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
import os

# Function to download model files from GitHub
def download_file_from_github(url, local_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            f.write(response.content)
    else:
        st.error(f"Failed to download {url}: Status code {response.status_code}")

# URLs of your model files on GitHub
repo_url = 'https://github.com/MarpakaPradeepSai/Instagram-Sentiment-Analysis-with-MiniLM/raw/main/minilm_model'  # Change to your actual repo URL
files = ['config.json', 'model.safetensors', 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.txt']

# Create model directory if it doesn't exist
model_dir = './minilm_model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Download each file
for file in files:
    download_file_from_github(f"{repo_url}/{file}", os.path.join(model_dir, file))

# Load tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.detach().numpy()

# Function to map probabilities to sentiment labels and emojis
def get_sentiment_label(probs):
    sentiment_mapping = ["Negative 😡", "Neutral 😐", "Positive 😊"]
    max_index = probs.argmax()
    return sentiment_mapping[max_index]

# Function to get background color based on sentiment
def get_background_color(label):
    if "Positive" in label:
        return "#C3E6CB"  # Softer green
    elif "Neutral" in label:
        return "#FFE8A1"  # Softer yellow
    else:
        return "#F5C6CB"  # Softer red

# Streamlit app
st.set_page_config(
    page_title="Sentiment Analysis with MiniLM",
    page_icon=":mag:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #F0F2F6;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 24px;
        cursor: pointer;
        font-size: 16px;
        transition: background-color 0.3s, transform 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
        transform: scale(1.05);
    }
    .prediction-box {
        border-radius: 25px;
        padding: 10px;
        text-align: center;
        font-size: 18px;
    }
    .center-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h1 style="font-size: 41px; text-align: center;">Instagram Sentiment Analysis with MiniLM</h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <img src="https://webcmstavtech.tav.aero/uploads/59f9875dc0e79a3594308ad3/static-pages/main-images/sentiment-analysis_1.jpg" alt="Sentiment Analysis" class="center-image" width="400">
    """,
    unsafe_allow_html=True
)

user_input = st.text_area("Enter text to analyze")

if st.button("Analyze"):
    if user_input:
        sentiment_probs = predict_sentiment(user_input)
        sentiment_label = get_sentiment_label(sentiment_probs[0])  # Get the label for the highest probability
        background_color = get_background_color(sentiment_label)  # Get the background color for the sentiment
        st.markdown(
            f"""
            <div style="background-color:{background_color}; padding: 10px; border-radius: 25px; text-align: center;" class="prediction-box">
                <h3>Sentiment: {sentiment_label}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.write("Please enter text to analyze.")
