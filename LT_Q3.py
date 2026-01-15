import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ----------------- Setup -----------------
st.set_page_config(page_title="Real-Time Webcam Classifier", layout="wide")
st.title("🚀 Real-Time Webcam Image Classification with ResNet18")

# Use CPU
device = torch.device("cpu")

# ----------------- Load Model -----------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.to(device).eval()
    return model

with st.spinner("Loading model..."):
    model = load_model()
    preprocess = models.ResNet18_Weights.DEFAULT.transforms()

# ----------------- Sidebar Settings -----------------
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of predictions", 1, 10, 5)
    show_details = st.checkbox("Show detailed analysis", value=False)

# ----------------- Webcam Capture via Streamlit -----------------
st.subheader("📸 Capture Image from Webcam")
uploaded_image = st.camera_input("Take a picture for classification")

if uploaded_image is not None:
    # Convert uploaded image to PIL
    image = Image.open(uploaded_image).convert("RGB")
    
    # Preprocess image for model
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Model prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output[0], dim=0)
        top_prob, top_indices = torch.topk(probabilities, top_k)
    
    # Display captured image
    st.image(image, caption="Captured Image", use_container_width=True)
    
    # Display Top-K predictions
    st.subheader(f"Top-{top_k} Predictions")
    data = []
    for i in range(top_k):
        label = models.ResNet18_Weights.DEFAULT.meta["categories"][top_indices[i]]
        prob = top_prob[i].item() * 100
        color = "green" if prob > 70 else "orange" if prob > 30 else "red"
        st.markdown(
            f"<span style='color:{color}; font-weight:bold'>{label}: {prob:.2f}%</span>",
            unsafe_allow_html=True
        )
        data.append({"Class": label, "Probability": prob})
    
    # Visualization in bar chart
    df = pd.DataFrame(data)
    st.bar_chart(df.set_index("Class"))
    
    # Detailed analysis if checkbox enabled
    if show_details:
        st.subheader("📊 Detailed Analysis")
        # Probability distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(df["Class"], df["Probability"], color="skyblue")
        ax.set_xlabel("Probability (%)")
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)
        
        # Tensor statistics
        st.write("**Image Tensor Statistics:**")
        st.json({
            "shape": list(input_tensor.shape),
            "mean": float(input_tensor.mean().item()),
            "std": float(input_tensor.std().item()),
            "range": [float(input_tensor.min().item()), float(input_tensor.max().item())]
        })

# ----------------- Footer -----------------
st.markdown("---")
st.markdown("**Note:** This app uses CPU; real-time inference may be slower than GPU.")
