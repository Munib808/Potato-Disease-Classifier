import streamlit as st
import streamlit.components.v1 as components
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
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── GLOBAL ─── */
.stApp {
    background: #06070A;
    font-family: 'Sora', sans-serif;
}

#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1000px; }

/* ─── ANIMATED BACKGROUND ─── */
.stApp > div:first-child::before {
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

/* ─── TYPOGRAPHY ─── */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Sora', sans-serif !important;
    color: #E8E6E1 !important;
}
p, span, li, .stMarkdown p {
    color: #A0A0A8 !important;
    font-family: 'Sora', sans-serif !important;
}

/* ─── HERO ─── */
.hero-container {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
    z-index: 2;
    animation: heroIn 1s ease-out;
}
@keyframes heroIn {
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
    width: 6px; height: 6px; border-radius: 50%;
    background: #10B981;
    animation: dotPulse 2s infinite;
}
@keyframes dotPulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,0.4); }
    50% { opacity: 0.6; box-shadow: 0 0 0 6px rgba(16,185,129,0); }
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
    flex-wrap: wrap;
}
.stat-item { text-align: center; }
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
.custom-divider {
    width: 60px; height: 2px;
    background: linear-gradient(90deg, transparent, #10B981, transparent);
    margin: 2rem auto;
    border-radius: 2px;
}

/* ─── UPLOAD HEADER ─── */
.upload-header {
    background: rgba(17, 17, 24, 0.7);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px 20px 0 0;
    padding: 1.25rem 2rem;
    display: flex;
    align-items: center;
    gap: 10px;
    position: relative;
    z-index: 2;
    animation: cardUp 0.8s ease-out 0.3s both;
    border-bottom: none;
}
.upload-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(16,185,129,0.3), transparent);
    border-radius: 20px 20px 0 0;
}
@keyframes cardUp {
    from { opacity: 0; transform: translateY(40px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ─── FILE UPLOADER ─── */
.stFileUploader {
    position: relative;
    z-index: 2;
}
.stFileUploader > div {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
.stFileUploader label { display: none !important; }

[data-testid="stFileUploaderDropzone"] {
    background: rgba(17, 17, 24, 0.7) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-top: 1px dashed rgba(255,255,255,0.08) !important;
    border-radius: 0 0 20px 20px !important;
    padding: 2.5rem 2rem !important;
    transition: all 0.4s ease !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(16, 185, 129, 0.3) !important;
    background: rgba(16, 185, 129, 0.03) !important;
}
[data-testid="stFileUploaderDropzone"] span {
    color: #A0A0A8 !important;
    font-family: 'Sora', sans-serif !important;
}
[data-testid="stFileUploaderDropzone"] small {
    color: #55535A !important;
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
}

/* ─── IMAGE ─── */
[data-testid="stImage"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stImage"] img { border-radius: 16px; }

/* ─── SECTION LABEL ─── */
.section-label {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.75rem;
}
.section-label span.icon { font-size: 14px; }
.section-label span.text {
    font-size: 0.75rem;
    font-weight: 500;
    color: #6B6B75 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ─── HIDE SUBHEADER ─── */
.stSubheader, [data-testid="stSubheader"] { display: none !important; }

/* ─── IFRAME ─── */
iframe { border: none !important; }

/* ─── FOOTER ─── */
.app-footer {
    text-align: center;
    padding: 3rem 0 2rem;
    margin-top: 2rem;
    border-top: 1px solid rgba(255,255,255,0.04);
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
    gap: 12px;
    margin-top: 0.75rem;
    flex-wrap: wrap;
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

@media (max-width: 768px) {
    .hero-title { font-size: 2rem; }
    .hero-stats { gap: 1.5rem; }
}
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
# RESULT CARD HTML (rendered via components.html to avoid raw HTML bug)
# ===============================
def build_result_html(info, confidence, probs):
    prob_colors = ["#F59E0B", "#EF4444", "#10B981"]
    prob_labels = ["Early Blight", "Late Blight", "Healthy"]

    prob_rows = ""
    for label, prob, color in zip(prob_labels, probs, prob_colors):
        prob_rows += f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
            <span style="font-size:13px;color:#A0A0A8;width:110px;flex-shrink:0;">{label}</span>
            <div style="flex:1;height:6px;background:rgba(255,255,255,0.04);border-radius:100px;overflow:hidden;">
                <div style="height:100%;width:{prob:.1f}%;background:{color};border-radius:100px;"></div>
            </div>
            <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#6B6B75;width:50px;text-align:right;flex-shrink:0;">{prob:.1f}%</span>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{
            font-family: 'Sora', sans-serif;
            background: transparent;
            color: #E8E6E1;
            padding: 4px;
        }}
        .card {{
            background: {info['bg']};
            border: 1px solid {info['border']};
            border-radius: 20px;
            padding: 1.75rem;
            animation: pop 0.7s ease-out;
        }}
        @keyframes pop {{
            from {{ opacity:0; transform:translateY(20px) scale(0.97); }}
            to {{ opacity:1; transform:translateY(0) scale(1); }}
        }}
        .hdr {{ display:flex; align-items:center; gap:14px; margin-bottom:1.25rem; padding-bottom:1.25rem; border-bottom:1px solid rgba(255,255,255,0.06); }}
        .ico {{ width:52px;height:52px;border-radius:14px;background:{info['bg']};border:1px solid {info['border']};display:flex;align-items:center;justify-content:center;font-size:24px; }}
        .cls {{ font-size:1.4rem;font-weight:600;color:#E8E6E1; }}
        .tag {{ font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:{info['color']};margin-top:3px; }}
        .cl {{ display:flex;justify-content:space-between;margin-bottom:8px; }}
        .clt {{ font-size:11px;color:#6B6B75;text-transform:uppercase;letter-spacing:0.08em;font-weight:500; }}
        .clv {{ font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:600;color:{info['color']}; }}
        .bar-bg {{ width:100%;height:8px;background:rgba(255,255,255,0.04);border-radius:100px;overflow:hidden;margin-bottom:1.25rem; }}
        .bar-fill {{ height:100%;border-radius:100px;background:linear-gradient(90deg,{info['color']}88,{info['color']});width:{confidence:.1f}%; }}
        .grid {{ display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:1.25rem; }}
        .gi {{ background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.04);border-radius:12px;padding:0.85rem; }}
        .gl {{ font-size:10px;color:#55535A;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px; }}
        .gv {{ font-size:14px;color:#E8E6E1;font-weight:500; }}
        .desc {{ font-size:13.5px;color:#A0A0A8;line-height:1.7;margin-bottom:1.25rem; }}
        .abox {{ background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:1.15rem; }}
        .ah {{ display:flex;align-items:center;gap:8px;margin-bottom:0.6rem; }}
        .at {{ font-size:11px;color:#E8E6E1;font-weight:500;text-transform:uppercase;letter-spacing:0.08em; }}
        .atxt {{ font-size:13px;color:#A0A0A8;line-height:1.7; }}
        .ps {{ margin-top:1.5rem;padding-top:1.25rem;border-top:1px solid rgba(255,255,255,0.06); }}
        .pt {{ font-size:10px;color:#55535A;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:1rem;font-weight:500; }}
    </style>
    </head>
    <body>
    <div class="card">
        <div class="hdr">
            <div class="ico">{info['icon']}</div>
            <div>
                <div class="cls">{info['label']}</div>
                <div class="tag">Detected condition</div>
            </div>
        </div>
        <div class="cl">
            <span class="clt">Confidence Score</span>
            <span class="clv">{confidence:.1f}%</span>
        </div>
        <div class="bar-bg"><div class="bar-fill"></div></div>
        <div class="grid">
            <div class="gi"><div class="gl">Severity</div><div class="gv">{info['severity']}</div></div>
            <div class="gi"><div class="gl">Model</div><div class="gv">MobileNet V2</div></div>
        </div>
        <div class="desc">{info['desc']}</div>
        <div class="abox">
            <div class="ah">
                <span style="font-size:14px;">💊</span>
                <span class="at">Recommended Action</span>
            </div>
            <div class="atxt">{info['action']}</div>
        </div>
        <div class="ps">
            <div class="pt">All Class Probabilities</div>
            {prob_rows}
        </div>
    </div>
    </body>
    </html>
    """
    return html

# ===============================
# SCANNING ANIMATION HTML
# ===============================
def build_scan_html(step=1):
    steps_data = [
        ("Upload", "done" if step >= 1 else ""),
        ("Preprocess", "active" if step == 1 else ("done" if step > 1 else "")),
        ("Classify", "active" if step == 2 else ("done" if step > 2 else "")),
        ("Result", "active" if step == 3 else ""),
    ]
    messages = [
        ("Analyzing leaf structure...", "Running MobileNet inference pipeline"),
        ("Running classification model...", "Extracting features from leaf patterns"),
        ("Finalizing results...", "Computing class probabilities"),
    ]
    msg = messages[min(step - 1, 2)]

    steps_html = ""
    for label, cls in steps_data:
        steps_html += f'<div class="ps {cls}"><span class="pd"></span><span>{label}</span></div>'

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
    <style>
        *{{margin:0;padding:0;box-sizing:border-box;}}
        body{{font-family:'Sora',sans-serif;background:transparent;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:280px;padding:2rem 1rem;}}
        .ring{{width:68px;height:68px;border-radius:50%;border:3px solid rgba(16,185,129,0.12);border-top-color:#10B981;animation:sp 1s linear infinite;display:flex;align-items:center;justify-content:center;font-size:26px;margin-bottom:1.25rem;}}
        @keyframes sp{{to{{transform:rotate(360deg);}}}}
        .st{{font-size:14px;color:#10B981;font-weight:500;text-align:center;}}
        .ss{{font-size:12px;color:#55535A;margin-top:0.35rem;text-align:center;}}
        .pr{{display:flex;justify-content:center;gap:1.5rem;margin-top:1.5rem;flex-wrap:wrap;}}
        .ps{{display:flex;align-items:center;gap:6px;font-size:11px;color:#3A3A42;}}
        .ps.active{{color:#10B981;}}
        .ps.done{{color:#8A8890;}}
        .pd{{width:7px;height:7px;border-radius:50%;background:#2A2A35;display:inline-block;}}
        .ps.done .pd{{background:#8A8890;}}
        .ps.active .pd{{background:#10B981;box-shadow:0 0 8px rgba(16,185,129,0.4);}}
    </style>
    </head>
    <body>
        <div class="ring">🔬</div>
        <div class="st">{msg[0]}</div>
        <div class="ss">{msg[1]}</div>
        <div class="pr">{steps_html}</div>
    </body>
    </html>
    """

# ===============================
# STREAMLIT UI
# ===============================

# ─── HERO ───
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

# ─── UPLOAD ───
st.markdown("""
<div class="upload-header">
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
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("""
        <div class="section-label">
            <span class="icon">🖼️</span>
            <span class="text">Input Image</span>
        </div>
        """, unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        # ─── SCANNING STEP 1 ───
        scan_slot = st.empty()
        with scan_slot.container():
            components.html(build_scan_html(step=1), height=320)

        time.sleep(0.8)

        # ─── SCANNING STEP 2 ───
        scan_slot.empty()
        with scan_slot.container():
            components.html(build_scan_html(step=2), height=320)

        # ─── ACTUAL PREDICTION (LOGIC UNTOUCHED) ───
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = predictions[0][predicted_index] * 100

        time.sleep(0.6)

        # ─── SHOW RESULT ───
        info = CLASS_DISPLAY[predicted_class]
        probs = predictions[0] * 100

        scan_slot.empty()
        with scan_slot.container():
            components.html(build_result_html(info, confidence, probs), height=700, scrolling=True)

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
