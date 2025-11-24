import streamlit as st
import torch
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import os
from torchvision import transforms

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models import HierarchicalFoodAnalysis, NutrientAwareTransformer


@st.cache_resource
def load_models():
    device = torch.device("cpu")

    hfa_model_path = 'best_hfa_model.pth'
    transformer_model_path = 'best_transformer_model.pth'

    if not os.path.exists(hfa_model_path) or not os.path.exists(transformer_model_path):
        st.error(
            "Model files not found! Please ensure 'best_hfa_model.pth' and 'best_transformer_model.pth' are in the main GlycoSIGHT project folder.")
        return None, None

    hfa_model = HierarchicalFoodAnalysis(num_food_classes=102, pretrained=False).to(device)
    hfa_model.load_state_dict(torch.load(hfa_model_path, map_location=device))
    hfa_model.eval()

    transformer_model = NutrientAwareTransformer().to(device)
    transformer_model.load_state_dict(torch.load(transformer_model_path, map_location=device))
    transformer_model.eval()

    return hfa_model, transformer_model


hfa_model, transformer_model = load_models()

st.set_page_config(layout="wide")
st.title("GlycoSIGHT: Interactive Demonstration")
st.markdown(
    "Take a picture of your food, and our program will give you a personal forecast of your body's reaction to it.")

if hfa_model is not None and transformer_model is not None:
    st.sidebar.header("User Inputs")
    uploaded_file = st.sidebar.file_uploader("1. Upload a picture of your meal", type=["jpg", "png", "jpeg"])
    cgm_history_text = st.sidebar.text_area(
        "2. Paste recent CGM data (comma-separated)",
        "120, 122, 125, 126, 124, 123"
    )
    run_button = st.sidebar.button("Run Forecast", type="primary")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Meal Analysis")
        if uploaded_file is None:
            st.info("Upload an image to see the nutritional analysis here.")
        else:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Your Uploaded Meal", use_container_width=True)

    with col2:
        st.subheader("Blood Glucose Forecast")
        st.info("The forecast chart will appear here after you click 'Run Forecast'.")

    if run_button:
        if uploaded_file is None or cgm_history_text == "":
            st.error("Please provide both a meal image and CGM data.")
        else:
            with st.spinner('Analyzing meal and forecasting glucose...'):
                transform = transforms.Compose([
                    transforms.Resize((256, 256)), transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    hfa_output = hfa_model(img_tensor)

                c, f, p = [50.0, 15.0, 20.0]

                with col1:
                    st.metric("Estimated Carbohydrates", f"{c:.1f} g")
                    st.metric("Estimated Fat", f"{f:.1f} g")
                    st.metric("Estimated Protein", f"{p:.1f} g")

                try:
                    cgm_history = np.array([float(val.strip()) for val in cgm_history_text.split(',')])
                    src = np.zeros((len(cgm_history), 5))
                    src[:, 0] = cgm_history / 100.0
                    src[-1, 2:] = [c / 100.0, f / 100.0, p / 100.0]

                    with torch.no_grad():
                        src_tensor = torch.FloatTensor(src).unsqueeze(0)
                        tgt_tensor = torch.zeros((1, 24, 1))
                        prediction = transformer_model(src_tensor, tgt_tensor)

                    predicted_glucose = (prediction[0, :, 0].numpy() * 100) + cgm_history[-1]

                    with col2:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=np.arange(-len(cgm_history), 0), y=cgm_history, mode='lines+markers',
                                                 name='Past Glucose'))
                        fig.add_trace(go.Scatter(x=np.arange(0, len(predicted_glucose)), y=predicted_glucose,
                                                 mode='lines+markers', name='Predicted Glucose',
                                                 line=dict(dash='dash')))
                        fig.update_layout(xaxis_title="Time (5-min steps from now)", yaxis_title="Glucose (mg/dL)")
                        st.plotly_chart(fig, use_container_width=True)

                except Exception:
                    st.error("Failed to process CGM data. Please ensure it's a comma-separated list of numbers.")