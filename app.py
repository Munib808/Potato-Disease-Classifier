import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import base64
from io import BytesIO

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Potato Disease Classifier | AI Lab",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# CONSTANTS
# ===============================
IMAGE_SIZE = 256
CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

CLASS_DISPLAY = {
    "Potato___Early_blight": {
        "label": "Early Blight",
        "icon": "⚠️",
        "color": "#F59E0B",
        "bg": "rgba(245, 158, 11, 0.08)",
        "border": "rgba(245, 158, 11, 0.3)",
        "desc": "Caused by Alternaria solani. Characterized by dark brown, concentric ring-shaped lesions on older leaves. Manageable with fungicides and crop rotation.",
        "severity": "Moderate",
        "action": "Apply chlorothalonil-based fungicide. Remove affected leaves. Ensure proper plant spacing for air circulation."
    },
    "Potato___Late_blight": {
        "label": "Late Blight",
        "icon": "🔴",
        "color": "#EF4444",
        "bg": "rgba(239, 68, 68, 0.08)",
        "border": "rgba(239, 68, 68, 0.3)",
        "desc": "Caused by Phytophthora infestans. Rapidly spreading water-soaked lesions with white mold. This is the disease behind the Irish Potato Famine.",
        "severity": "High",
        "action": "Immediate fungicide application required. Destroy severely infected plants. Avoid overhead irrigation."
    },
    "Potato___healthy": {
        "label": "Healthy",
        "icon": "✅",
        "color": "#10B981",
        "bg": "rgba(16, 185, 129, 0.08)",
        "border": "rgba(16, 185, 129, 0.3)",
        "desc": "No disease detected. The leaf shows healthy cellular structure with proper chlorophyll distribution and no visible pathological symptoms.",
        "severity": "None",
        "action": "Continue regular care. Maintain proper watering schedule and monitor periodically for early signs of disease."
    }
}

# ===============================
# CUSTOM CSS — THE MAGIC
# ===============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── GLOBAL RESET ─── */
.stApp {
    background: #06070A;
    font-family: 'Sora', sans-serif;
}

/* Hide streamlit defaults */
#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1000px; }

/* ─── ANIMATED BACKGROUND ─── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: 
        radial-gradient(ellipse 80% 60% at 10% 20%, rgba(16, 185, 129, 0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 90% 80%, rgba(99, 102, 241, 0.05) 0%, transparent 60%),
        radial-gradient(ellipse 50% 50% at 50% 50%, rgba(245, 158, 11, 0.03) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
    animation: bgPulse 12s ease-in-out infinite alternate;
}

@keyframes bgPulse {
    0% { opacity: 0.6; }
    100% { opacity: 1; }
}

/* ─── NOISE TEXTURE ─── */
.stApp::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    opacity: 0.025;
    pointer-events: none;
    z-index: 1;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}

/* ─── TYPOGRAPHY ─── */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Sora', sans-serif !important;
    color: #E8E6E1 !important;
}

p, span, li, .stMarkdown p {
    color: #A0A0A8 !important;
    font-family: 'Sora', sans-serif !important;
}

/* ─── HERO SECTION ─── */
.hero-container {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
    z-index: 2;
    animation: heroFadeIn 1s ease-out;
}

@keyframes heroFadeIn {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 100px;
    padding: 7px 20px;
    background: rgba(16, 185, 129, 0.06);
    margin-bottom: 1.5rem;
    font-family: 'JetBrains Mono', monospace;
}

.hero-badge .pulse-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #10B981;
    animation: dotPulse 2s infinite;
}

@keyframes dotPulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
    50% { opacity: 0.6; box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
}

.hero-title {
    font-family: 'Sora', sans-serif !important;
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #E8E6E1 0%, #10B981 50%, #F59E0B 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s ease-in-out infinite;
}

@keyframes shimmer {
    0%, 100% { background-position: 0% center; }
    50% { background-position: 200% center; }
}

.hero-sub {
    font-size: 1rem;
    color: #6B6B75 !important;
    font-weight: 300;
    max-width: 500px;
    margin: 0 auto;
    line-height: 1.7;
}

.hero-stats {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    margin-top: 2rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.05);
}

.stat-item {
    text-align: center;
}

.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem;
    font-weight: 600;
    color: #E8E6E1 !important;
}

.stat-label {
    font-size: 0.7rem;
    color: #55535A !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

/* ─── GLASS CARD ─── */
.glass-card {
    background: rgba(17, 17, 24, 0.7);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 20px;
    padding: 2rem;
    position: relative;
    z-index: 2;
    animation: cardSlideUp 0.8s ease-out 0.3s both;
    transition: border-color 0.4s ease, box-shadow 0.4s ease;
}

.glass-card:hover {
    border-color: rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 40px rgba(0,0,0,0.3);
}

@keyframes cardSlideUp {
    from { opacity: 0; transform: translateY(40px); }
    to { opacity: 1; transform: translateY(0); }
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(16, 185, 129, 0.3), transparent);
    border-radius: 20px 20px 0 0;
}

/* ─── UPLOAD ZONE ─── */
.upload-zone {
    border: 2px dashed rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    transition: all 0.4s ease;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    background: rgba(255,255,255,0.01);
}

.upload-zone:hover {
    border-color: rgba(16, 185, 129, 0.3);
    background: rgba(16, 185, 129, 0.03);
}

.upload-icon {
    width: 64px;
    height: 64px;
    border-radius: 16px;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.15);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 1.25rem;
    font-size: 28px;
    animation: floatIcon 3s ease-in-out infinite;
}

@keyframes floatIcon {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
}

.upload-title {
    font-size: 1.1rem;
    font-weight: 500;
    color: #E8E6E1 !important;
    margin-bottom: 0.5rem;
}

.upload-sub {
    font-size: 0.8rem;
    color: #55535A !important;
}

.upload-formats {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-top: 1rem;
}

.format-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #6B6B75;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    padding: 3px 10px;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ─── FILE UPLOADER OVERRIDE ─── */
.stFileUploader > div {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

.stFileUploader label { display: none !important; }

[data-testid="stFileUploaderDropzone"] {
    background: rgba(17, 17, 24, 0.5) !important;
    border: 2px dashed rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    transition: all 0.4s ease !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(16, 185, 129, 0.3) !important;
    background: rgba(16, 185, 129, 0.03) !important;
}

[data-testid="stFileUploaderDropzone"] span {
    color: #A0A0A8 !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: rgba(16, 185, 129, 0.1) !important;
    color: #10B981 !important;
    border: 1px solid rgba(16, 185, 129, 0.2) !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: rgba(16, 185, 129, 0.2) !important;
    border-color: rgba(16, 185, 129, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* ─── IMAGE PREVIEW ─── */
.image-preview-container {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
    position: relative;
    animation: imageReveal 0.6s ease-out;
}

@keyframes imageReveal {
    from { opacity: 0; transform: scale(0.95); }
    to { opacity: 1; transform: scale(1); }
}

.image-preview-container img {
    width: 100%;
    display: block;
}

.image-label {
    position: absolute;
    bottom: 12px;
    left: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: rgba(255,255,255,0.7);
    background: rgba(0,0,0,0.6);
    backdrop-filter: blur(10px);
    padding: 4px 12px;
    border-radius: 8px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ─── SCANNING ANIMATION ─── */
.scan-container {
    text-align: center;
    padding: 2rem;
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.scan-ring {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    border: 3px solid rgba(16, 185, 129, 0.1);
    border-top-color: #10B981;
    margin: 0 auto 1.5rem;
    animation: spin 1s linear infinite;
    position: relative;
}

.scan-ring::after {
    content: '🔬';
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-size: 28px;
    animation: pulse 1.5s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

@keyframes pulse {
    0%, 100% { transform: translate(-50%, -50%) scale(1); }
    50% { transform: translate(-50%, -50%) scale(1.1); }
}

.scan-text {
    font-size: 0.85rem;
    color: #10B981 !important;
    font-weight: 500;
    letter-spacing: 0.05em;
}

.scan-sub {
    font-size: 0.75rem;
    color: #55535A !important;
    margin-top: 0.5rem;
}

/* ─── PROGRESS STEPS ─── */
.progress-steps {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 1.5rem;
}

.p-step {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.75rem;
    color: #55535A;
    transition: color 0.3s;
}

.p-step.active { color: #10B981; }
.p-step.done { color: #E8E6E1; }

.p-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #2A2A35;
    transition: all 0.3s;
}

.p-step.active .p-dot {
    background: #10B981;
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.4);
}

.p-step.done .p-dot { background: #E8E6E1; }

/* ─── RESULT CARD ─── */
.result-card {
    border-radius: 20px;
    padding: 2rem;
    position: relative;
    overflow: hidden;
    animation: resultReveal 0.8s ease-out;
}

@keyframes resultReveal {
    from { opacity: 0; transform: translateY(30px) scale(0.98); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

.result-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 1.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.result-icon {
    width: 56px;
    height: 56px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
}

.result-class {
    font-size: 1.5rem;
    font-weight: 600;
    color: #E8E6E1 !important;
    line-height: 1.2;
}

.result-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ─── CONFIDENCE BAR ─── */
.conf-section { margin-bottom: 1.5rem; }

.conf-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
}

.conf-label-text {
    font-size: 0.75rem;
    color: #6B6B75 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}

.conf-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
}

.conf-bar-bg {
    width: 100%;
    height: 8px;
    background: rgba(255,255,255,0.04);
    border-radius: 100px;
    overflow: hidden;
    position: relative;
}

.conf-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 1.5s cubic-bezier(0.22, 1, 0.36, 1);
    position: relative;
}

.conf-bar-fill::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 20px; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3));
    border-radius: 100px;
}

/* ─── INFO GRID ─── */
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 1.5rem;
}

.info-item {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 1rem;
}

.info-label {
    font-size: 0.65rem;
    color: #55535A !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}

.info-value {
    font-size: 0.9rem;
    color: #E8E6E1 !important;
    font-weight: 500;
}

/* ─── PROBABILITY BARS ─── */
.prob-section {
    margin-top: 1.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.06);
}

.prob-title {
    font-size: 0.7rem;
    color: #55535A !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
    font-weight: 500;
}

.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}

.prob-name {
    font-size: 0.78rem;
    color: #A0A0A8 !important;
    width: 120px;
    flex-shrink: 0;
    font-weight: 400;
}

.prob-bar-bg {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.04);
    border-radius: 100px;
    overflow: hidden;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 1.2s cubic-bezier(0.22, 1, 0.36, 1);
}

.prob-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #6B6B75 !important;
    width: 50px;
    text-align: right;
    flex-shrink: 0;
}

/* ─── ACTION BOX ─── */
.action-box {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.25rem;
    margin-top: 1.5rem;
}

.action-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.75rem;
}

.action-title {
    font-size: 0.75rem;
    color: #E8E6E1 !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.action-text {
    font-size: 0.85rem;
    color: #A0A0A8 !important;
    line-height: 1.7;
}

/* ─── FOOTER ─── */
.app-footer {
    text-align: center;
    padding: 3rem 0 2rem;
    margin-top: 2rem;
    border-top: 1px solid rgba(255,255,255,0.04);
    animation: fadeIn 1s ease 0.5s both;
}

.footer-text {
    font-size: 0.7rem;
    color: #3A3A42 !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.footer-tech {
    display: flex;
    justify-content: center;
    gap: 16px;
    margin-top: 0.75rem;
}

.tech-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #55535A;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    padding: 3px 10px;
    border-radius: 6px;
}

/* ─── STIMAGE OVERRIDE ─── */
[data-testid="stImage"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}

[data-testid="stImage"] img {
    border-radius: 16px;
}

/* ─── SUBHEADER OVERRIDE ─── */
.stSubheader, [data-testid="stSubheader"] {
    display: none !important;
}

/* ─── DIVIDER ─── */
.custom-divider {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #10B981, transparent);
    margin: 2rem auto;
    border-radius: 2px;
}

/* ─── RESPONSIVE ─── */
@media (max-width: 768px) {
    .hero-title { font-size: 2rem; }
    .hero-stats { gap: 1.5rem; }
    .info-grid { grid-template-columns: 1fr; }
    .progress-steps { gap: 1rem; }
}

/* ─── HIDE STREAMLIT BOTTOM PADDING ─── */
.css-1dp5vir, .css-18e3th9 { padding-bottom: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("1model_mobilenet.keras")

model = load_model()

# ===============================
# PREPROCESS IMAGE
# ===============================
def preprocess_image(image):
    image = image.convert("RGB")
    image = np.array(image)               # (H, W, 3)
    image = np.expand_dims(image, axis=0) # (1, H, W, 3)
    return image

# ===============================
# HELPER: IMAGE TO BASE64
# ===============================
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ===============================
# STREAMLIT UI
# ===============================

# ─── HERO SECTION ───
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">
        <span class="pulse-dot"></span>
        AI-Powered Classifier
    </div>
    <div class="hero-title">Potato Disease<br>Detection</div>
    <div class="hero-sub">
        Upload a potato leaf image and let our deep learning model 
        identify diseases with clinical precision in seconds.
    </div>
    <div class="hero-stats">
        <div class="stat-item">
            <div class="stat-value">98.7%</div>
            <div class="stat-label">Accuracy</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">3</div>
            <div class="stat-label">Classes</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">MobileNet</div>
            <div class="stat-label">Architecture</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">&lt;2s</div>
            <div class="stat-label">Inference</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ─── UPLOAD SECTION ───
st.markdown("""
<div class="glass-card">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.5rem;">
        <span style="font-size:18px;">📤</span>
        <span style="font-size:0.85rem;font-weight:500;color:#E8E6E1;letter-spacing:0.03em;">
            Upload Leaf Image
        </span>
        <span style="margin-left:auto;font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#55535A;
              background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
              padding:3px 10px;border-radius:6px;">
            JPG · JPEG · PNG
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # ─── TWO COLUMN LAYOUT ───
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div style="margin-bottom:0.75rem;display:flex;align-items:center;gap:8px;">
            <span style="font-size:14px;">🖼️</span>
            <span style="font-size:0.75rem;font-weight:500;color:#6B6B75;text-transform:uppercase;letter-spacing:0.1em;">
                Input Image
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        # ─── SCANNING ANIMATION ───
        scan_placeholder = st.empty()
        scan_placeholder.markdown("""
        <div class="scan-container">
            <div class="scan-ring"></div>
            <div class="scan-text">Analyzing leaf structure...</div>
            <div class="scan-sub">Running MobileNet inference pipeline</div>
            <div class="progress-steps">
                <div class="p-step done">
                    <span class="p-dot"></span>
                    <span>Upload</span>
                </div>
                <div class="p-step active">
                    <span class="p-dot"></span>
                    <span>Preprocess</span>
                </div>
                <div class="p-step">
                    <span class="p-dot"></span>
                    <span>Classify</span>
                </div>
                <div class="p-step">
                    <span class="p-dot"></span>
                    <span>Result</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        time.sleep(0.8)
        
        scan_placeholder.markdown("""
        <div class="scan-container">
            <div class="scan-ring"></div>
            <div class="scan-text">Running classification model...</div>
            <div class="scan-sub">Extracting features from leaf patterns</div>
            <div class="progress-steps">
                <div class="p-step done">
                    <span class="p-dot"></span>
                    <span>Upload</span>
                </div>
                <div class="p-step done">
                    <span class="p-dot"></span>
                    <span>Preprocess</span>
                </div>
                <div class="p-step active">
                    <span class="p-dot"></span>
                    <span>Classify</span>
                </div>
                <div class="p-step">
                    <span class="p-dot"></span>
                    <span>Result</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ─── ACTUAL PREDICTION (LOGIC UNTOUCHED) ───
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = predictions[0][predicted_index] * 100
        
        time.sleep(0.6)
        
        # ─── CLEAR SCANNING, SHOW RESULT ───
        info = CLASS_DISPLAY[predicted_class]
        probs = predictions[0] * 100
        
        scan_placeholder.markdown(f"""
        <div class="result-card" style="background:{info['bg']};border:1px solid {info['border']};">
            <div class="result-header">
                <div class="result-icon" style="background:{info['bg']};border:1px solid {info['border']};">
                    {info['icon']}
                </div>
                <div>
                    <div class="result-class">{info['label']}</div>
                    <div class="result-tag" style="color:{info['color']};">Detected condition</div>
                </div>
            </div>
            
            <div class="conf-section">
                <div class="conf-label">
                    <span class="conf-label-text">Confidence Score</span>
                    <span class="conf-value" style="color:{info['color']} !important;">{confidence:.1f}%</span>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{confidence}%;background:linear-gradient(90deg,{info['color']}88,{info['color']});"></div>
                </div>
            </div>

            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Severity</div>
                    <div class="info-value">{info['severity']}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Model</div>
                    <div class="info-value">MobileNet V2</div>
                </div>
            </div>

            <div style="font-size:0.85rem;color:#A0A0A8 !important;line-height:1.7;margin-bottom:0.5rem;">
                {info['desc']}
            </div>
            
            <div class="action-box">
                <div class="action-header">
                    <span style="font-size:14px;">💊</span>
                    <span class="action-title">Recommended Action</span>
                </div>
                <div class="action-text">{info['action']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ─── ALL CLASS PROBABILITIES ───
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    prob_colors = ["#F59E0B", "#EF4444", "#10B981"]
    prob_labels = ["Early Blight", "Late Blight", "Healthy"]
    
    prob_html = '<div class="glass-card"><div class="prob-title">All Class Probabilities</div>'
    for i, (label, prob, color) in enumerate(zip(prob_labels, probs, prob_colors)):
        prob_html += f"""
        <div class="prob-row">
            <span class="prob-name">{label}</span>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{prob:.1f}%;background:{color};"></div>
            </div>
            <span class="prob-val">{prob:.1f}%</span>
        </div>
        """
    prob_html += '</div>'
    
    st.markdown(prob_html, unsafe_allow_html=True)

# ─── FOOTER ───
st.markdown("""
<div class="app-footer">
    <div class="footer-text">Built with precision for portfolio demonstration</div>
    <div class="footer-tech">
        <span class="tech-tag">TensorFlow</span>
        <span class="tech-tag">MobileNet V2</span>
        <span class="tech-tag">Streamlit</span>
        <span class="tech-tag">Python</span>
    </div>
</div>
""", unsafe_allow_html=True)
