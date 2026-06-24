# =============================================================================
# Visual Question Answering (VQA) Mini Project
# B.Sc. Data Science - Computer Vision Project
# =============================================================================
# Description:
#   This application allows users to upload an image and ask natural language
#   questions about it. A pre-trained Hugging Face VQA model analyzes the
#   image and question together to generate an intelligent answer.
#
# Model Used: Salesforce/blip-vqa-base (BLIP - Bootstrapped Language-Image Pre-training)
# Framework: Streamlit (for the interactive web UI)
# =============================================================================

# --- Import Required Libraries ---
import streamlit as st                          # Web app framework
from PIL import Image                           # Image loading and processing
from transformers import pipeline               # Hugging Face model pipeline
import torch                                    # PyTorch (backend for the model)
import time                                     # For simulating load delays
import io                                       # For handling byte streams

# =============================================================================
# PAGE CONFIGURATION
# Must be the first Streamlit command in the script
# =============================================================================
st.set_page_config(
    page_title="VQA - Visual Question Answering",
    page_icon="🔍",
    layout="wide",                              # Use the full width of the page
    initial_sidebar_state="expanded"            # Show sidebar by default
)

# =============================================================================
# CUSTOM CSS STYLING
# Makes the app look professional and modern
# =============================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main title styling */
    .main-title {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(15, 52, 96, 0.3);
    }

    .main-title h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-title p {
        font-size: 1rem;
        opacity: 0.8;
        margin: 0.5rem 0 0 0;
    }

    /* Answer card */
    .answer-card {
        background: linear-gradient(135deg, #e0f7fa 0%, #e8f5e9 100%);
        border-left: 5px solid #00897b;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,137,123,0.1);
    }

    .answer-card h3 {
        color: #00695c;
        margin: 0 0 0.5rem 0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .answer-text {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0;
    }

    /* Confidence badge */
    .confidence-badge {
        display: inline-block;
        background: #0f3460;
        color: white;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.8rem;
    }

    /* Info cards in sidebar */
    .info-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.8rem 0;
        border: 1px solid #e9ecef;
    }

    /* Step indicator */
    .step-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #0f3460;
    }

    /* Upload area */
    .upload-area {
        border: 2px dashed #0f3460;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #f0f4ff;
        margin-bottom: 1rem;
    }

    /* Question box */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.8rem 1rem;
        font-size: 1rem;
        transition: border-color 0.3s;
    }

    .stTextInput > div > div > input:focus {
        border-color: #0f3460;
        box-shadow: 0 0 0 3px rgba(15, 52, 96, 0.1);
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #0f3460, #16213e);
        color: white;
        border: none;
        padding: 0.75rem 2.5rem;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
        cursor: pointer;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(15, 52, 96, 0.3);
    }

    /* Error box */
    .error-box {
        background: #fff5f5;
        border-left: 4px solid #e53e3e;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        color: #c53030;
        font-weight: 500;
    }

    /* Warning box */
    .warning-box {
        background: #fffbeb;
        border-left: 4px solid #d69e2e;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        color: #744210;
        font-weight: 500;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MODEL LOADING FUNCTION
# @st.cache_resource caches the model so it's only loaded ONCE
# (not re-loaded every time the user interacts with the app)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_vqa_model():
    """
    Load the pre-trained BLIP VQA model from Hugging Face.

    BLIP (Bootstrapped Language-Image Pre-training) is a state-of-the-art
    Vision-Language model that can understand both images and text questions.

    Returns:
        pipeline: A Hugging Face pipeline object for visual question answering.
    """
    # Automatically use GPU if available, otherwise fall back to CPU
    device = 0 if torch.cuda.is_available() else -1

    # Load the VQA pipeline with the BLIP model
    # This will download the model weights on first run (~900MB)
    vqa_pipe = pipeline(
        task="visual-question-answering",
        model="Salesforce/blip-vqa-base",   # Pre-trained BLIP VQA model
        device=device
    )
    return vqa_pipe


# =============================================================================
# ANSWER GENERATION FUNCTION
# =============================================================================
def get_vqa_answer(model_pipeline, image: Image.Image, question: str) -> dict:
    """
    Use the VQA model to answer a question about an image.

    Args:
        model_pipeline: The loaded Hugging Face VQA pipeline.
        image (PIL.Image): The uploaded image (in PIL format).
        question (str): The user's natural language question.

    Returns:
        dict: Contains 'answer' (str) and 'score' (float confidence 0–1).
    """
    # Run the model — it analyzes both image and question simultaneously
    results = model_pipeline(image, question, top_k=1)

    # The pipeline returns a list; we take the top (best) prediction
    top_result = results[0]
    return {
        "answer": top_result["answer"],
        "score": top_result["score"]
    }


# =============================================================================
# SIDEBAR — Project Information & How-To Guide
# =============================================================================
with st.sidebar:
    st.markdown("## 🔬 About This Project")
    st.markdown("""
    <div class='info-card'>
        <strong>📌 Project:</strong> Visual Question Answering<br>
        <strong>📚 Course:</strong> B.Sc. Data Science<br>
        <strong>🏷️ Domain:</strong> Computer Vision + NLP
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🧠 Model Info")
    st.markdown("""
    <div class='info-card'>
        <strong>Model:</strong> BLIP-VQA-Base<br>
        <strong>Source:</strong> Salesforce / Hugging Face<br>
        <strong>Parameters:</strong> ~250 Million<br>
        <strong>Task:</strong> Visual Question Answering
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📋 How to Use")

    steps = [
        ("1️⃣", "Upload an image (JPG, JPEG, PNG)"),
        ("2️⃣", "The image will appear on screen"),
        ("3️⃣", "Type a question about the image"),
        ("4️⃣", "Click 'Get Answer'"),
        ("5️⃣", "View the AI-generated answer"),
    ]

    for icon, text in steps:
        st.markdown(f"""
        <div class='step-card'>
            {icon} {text}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 💡 Sample Questions")
    st.markdown("""
    - *What color is the car?*
    - *How many people are in the image?*
    - *What is the weather like?*
    - *What animal is in the photo?*
    - *Is it day or night?*
    - *What sport is being played?*
    """)


# =============================================================================
# MAIN PAGE — Title Header
# =============================================================================
st.markdown("""
<div class='main-title'>
    <h1>🔍 Visual Question Answering</h1>
    <p>Upload an image · Ask a question · Get an AI-powered answer</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# MODEL LOADING — Show spinner while model loads
# =============================================================================
with st.spinner("⏳ Loading BLIP VQA model (first load may take 1–2 minutes)..."):
    try:
        vqa_model = load_vqa_model()
        st.success("✅ Model loaded successfully and ready to use!")
    except Exception as e:
        # If model fails to load, show an error and stop the app
        st.markdown(f"""
        <div class='error-box'>
            ❌ <strong>Model loading failed:</strong> {str(e)}<br>
            Please ensure you have a working internet connection and transformers is installed.
        </div>
        """, unsafe_allow_html=True)
        st.stop()   # Halt execution — no point continuing without the model


# =============================================================================
# TWO-COLUMN LAYOUT
# Left column: Image upload + display
# Right column: Question input + answer display
# =============================================================================
col_left, col_right = st.columns([1, 1], gap="large")

# -----------------------------------------------
# LEFT COLUMN — Image Upload
# -----------------------------------------------
with col_left:
    st.markdown("### 📷 Step 1: Upload Your Image")

    # File uploader widget — accepts JPG, JPEG, PNG
    uploaded_file = st.file_uploader(
        label="Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG. Max size: 200MB."
    )

    # Store the image in session state so it persists between interactions
    if uploaded_file is not None:
        # Read and convert the uploaded file to a PIL Image (RGB mode)
        image_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Save to session state so it's accessible in the right column
        st.session_state["uploaded_image"] = pil_image

        # Display the uploaded image on screen
        st.image(
            pil_image,
            caption=f"📸 {uploaded_file.name}  |  Size: {pil_image.size[0]}×{pil_image.size[1]} px",
            use_container_width=True
        )

        # Show image metadata
        st.markdown(f"""
        <div class='info-card'>
            📁 <strong>File:</strong> {uploaded_file.name}<br>
            📐 <strong>Dimensions:</strong> {pil_image.size[0]} × {pil_image.size[1]} pixels<br>
            🎨 <strong>Mode:</strong> {pil_image.mode}
        </div>
        """, unsafe_allow_html=True)

    else:
        # No image uploaded yet — show placeholder instructions
        st.markdown("""
        <div class='upload-area'>
            <p style='font-size:3rem;margin:0'>🖼️</p>
            <p style='color:#555;font-size:1rem;'>Drag & drop your image here<br>or click <strong>Browse files</strong></p>
            <p style='color:#999;font-size:0.85rem;'>JPG · JPEG · PNG supported</p>
        </div>
        """, unsafe_allow_html=True)

        # Clear session state if a previous image existed
        if "uploaded_image" in st.session_state:
            del st.session_state["uploaded_image"]


# -----------------------------------------------
# RIGHT COLUMN — Question & Answer
# -----------------------------------------------
with col_right:
    st.markdown("### ❓ Step 2: Ask a Question")

    # Text input for the user's question
    user_question = st.text_input(
        label="Type your question here",
        placeholder="e.g. What color is the dog? How many people are there?",
        help="Ask anything about the image — objects, colors, counts, actions, etc."
    )

    # Spacer
    st.markdown("<br>", unsafe_allow_html=True)

    # Analyze button — triggers inference
    analyze_button = st.button("🔍 Get Answer", use_container_width=True)

    # -----------------------------------------------
    # INFERENCE LOGIC — Run when button is clicked
    # -----------------------------------------------
    if analyze_button:

        # --- VALIDATION: Check if image is uploaded ---
        if "uploaded_image" not in st.session_state or st.session_state["uploaded_image"] is None:
            st.markdown("""
            <div class='error-box'>
                ❌ <strong>No image uploaded!</strong><br>
                Please upload an image in the left panel before asking a question.
            </div>
            """, unsafe_allow_html=True)

        # --- VALIDATION: Check if question is entered ---
        elif not user_question.strip():
            st.markdown("""
            <div class='warning-box'>
                ⚠️ <strong>Question is empty!</strong><br>
                Please type a question about the image before clicking 'Get Answer'.
            </div>
            """, unsafe_allow_html=True)

        else:
            # Both image and question are present — proceed with inference
            try:
                with st.spinner("🤖 AI is analyzing your image and question..."):
                    # Retrieve the stored image from session state
                    image_to_analyze = st.session_state["uploaded_image"]

                    # Run the VQA model
                    result = get_vqa_answer(vqa_model, image_to_analyze, user_question)

                    # Extract answer and confidence score
                    answer_text = result["answer"].capitalize()
                    confidence = result["score"]
                    confidence_pct = round(confidence * 100, 1)

                    # Brief pause for UX (makes the transition feel smooth)
                    time.sleep(0.3)

                # --- Display the Answer ---
                st.markdown(f"""
                <div class='answer-card'>
                    <h3>💬 AI Answer</h3>
                    <p class='answer-text'>{answer_text}</p>
                    <div class='confidence-badge'>
                        Confidence: {confidence_pct}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Show additional context based on confidence level
                st.markdown("<br>", unsafe_allow_html=True)
                if confidence >= 0.8:
                    st.success("✅ High confidence — the model is very sure about this answer.")
                elif confidence >= 0.5:
                    st.info("ℹ️ Medium confidence — the answer is likely correct but may vary.")
                else:
                    st.warning("⚠️ Low confidence — consider rephrasing your question.")

                # Display the question-answer pair for reference
                st.markdown("---")
                st.markdown("##### 📝 Summary")
                st.markdown(f"**Question:** {user_question}")
                st.markdown(f"**Answer:** {answer_text}")
                st.markdown(f"**Confidence Score:** {confidence_pct}%")

            except Exception as e:
                # Handle any inference errors gracefully
                st.markdown(f"""
                <div class='error-box'>
                    ❌ <strong>Inference Error:</strong> {str(e)}<br>
                    Try uploading a clearer image or rephrasing your question.
                </div>
                """, unsafe_allow_html=True)

    else:
        # Show placeholder before the button is clicked
        if "uploaded_image" in st.session_state:
            st.markdown("""
            <div style='text-align:center; color:#888; padding:3rem 1rem;
                        border:2px dashed #ddd; border-radius:12px; margin-top:1rem;'>
                <p style='font-size:2rem;margin:0'>🤖</p>
                <p>Enter a question and click <strong>Get Answer</strong><br>to see the AI response here.</p>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# FOOTER — Project info
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#999; font-size:0.85rem; padding:1rem 0;'>
    🎓 B.Sc. Data Science · Computer Vision Mini Project<br>
    Powered by <strong>BLIP VQA</strong> (Salesforce) · Built with <strong>Streamlit</strong> &
    <strong>Hugging Face Transformers</strong>
</div>
""", unsafe_allow_html=True)
