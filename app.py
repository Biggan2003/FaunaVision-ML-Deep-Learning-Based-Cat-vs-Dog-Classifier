import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# --- Page Configuration & Icon ---
st.set_page_config(
    page_title="FaunaVision ML: Cats & Dogs Classifier",
    page_icon="🐾",
    layout="centered"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* Warning Box styling */
    .warning-box {
        background-color: #FFF3CD;
        border: 1px solid #FFEBA0;
        padding: 15px;
        border-radius: 8px;
        color: #856404;
        font-weight: bold;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# --- Header & Description ---
st.markdown("<h1 style='text-align: center;'>FaunaVision ML 🐾</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1em; '>Deep Learning Based Cat vs Dog Classifier Built with PyTorch & CNN</p>", unsafe_allow_html=True)
st.markdown("<hr style='margin: 10px 0 25px 0;'>", unsafe_allow_html=True)

# --- User Warning (English Version) ---
st.markdown("""
<div class='warning-box'>
    ⚠️ NOTE: This Model is Trained strictly to classify CATS and DOGS. Uploading other images (e.g., humans, objects, or text documents) will return incorrect classifications.".
</div>
""", unsafe_allow_html=True)

# --- Main Layout Sections ---
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<div style='font-size: 120px; text-align: center;'>🐱🐶</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7F8C8D; font-style: italic;'>'Waiting for Image...'</p>", unsafe_allow_html=True)

with col2:
    st.markdown("### How to Use?")
    st.write("1. Upload a clear image from your device or use your camera.")
    st.write("2. The model will analyze the features in real-time.")
    st.write("3. Get the prediction result along with the confidence score instantly.")

# --- Model Architecture ---
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        if hasattr(self, 'bn4'):
            x = self.pool(F.relu(self.bn4(self.conv4(x))))
        else:
            x = self.pool(F.relu(self.conv4(x)))
            
        x = x.view(-1, 128 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CatDogCNN()
    try:
        model.load_state_dict(torch.load('cat_dog_cnn.pth', map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None
    return model, device

# --- Image Transformations ---
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model, device = load_model()

if model is None:
    st.stop()

# --- File Uploader & Camera Inputs ---
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
camera_input = st.camera_input("Or take a photo...")

if uploaded_file is not None or camera_input is not None:
    image_file = uploaded_file if uploaded_file is not None else camera_input
    image = Image.open(image_file).convert('RGB')
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner("Analyzing image... Please wait!"):
        input_tensor = img_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            conf, predicted = torch.max(probabilities, 1)
        
        confidence_score = conf.item() * 100
        
        # 🛡️ Confidence Threshold Check
        if confidence_score < 80.0:
            st.warning("⚠️ AI is uncertain! Please upload a clearer image of a Cat or a Dog.")
            st.info(f"**Model's Dillema:** Match confidence is only {confidence_score:.2f}%, which is below the acceptable threshold.")
        else:
            classes = ['Cat', 'Dog']
            result = classes[predicted.item()]
            
            # --- Results Display ---
            st.success(f"**Prediction:** It's a **{result}**!")
            st.info(f"**Confidence Level:** {confidence_score:.2f}%")
            
            # --- Footer Section (Creator Credit) ---
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 0.9em; color: #888888;'>"
    "Created by <a href='https://www.linkedin.com/in/g-m-biggan-371956305/' target='_blank' style='font-weight: bold; text-decoration: none;'>G. M Biggan</a>"
    "</p>", 
    unsafe_allow_html=True
)