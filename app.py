import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ===========================================================
# 1. LOAD ORIGINAL FINBERT (cached)
# ===========================================================
@st.cache_resource
def load_finbert():
    MODEL_NAME = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, finbert_model = load_finbert()


def predict_finbert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = finbert_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
    prob_compliant = probs[0]
    prob_non_compliant = probs[1]

    label = "Non-Compliant" if prob_non_compliant > 0.5 else "Compliant"

    return {
        "prediction": label,
        "prob_non_compliant": float(prob_non_compliant)
    }


# ===========================================================
# 2. STREAMLIT PAGE CONFIG
# ===========================================================
st.set_page_config(
    page_title="AML Narrative Classification System",
    layout="wide",
    page_icon="🔍"
)


# ===========================================================
# 3. CUSTOM STYLING
# ===========================================================
st.markdown("""
    <style>
        .title {
            font-size: 42px !important;
            color: #4F8BF9;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0px;
        }
        .subtitle {
            font-size: 19px !important;
            color: #2E4053;
            text-align: center;
            margin-top: -10px;
        }
        .result-box {
            background-color: #F4F6F7;
            padding: 20px;
            border-radius: 10px;
            border-left: 6px solid #4F8BF9;
            margin-bottom: 20px;
        }
        .section-title {
            font-size: 22px;
            font-weight: 600;
            color: #4F8BF9;
            margin-top: 20px;
        }
        .footer {
            font-size: 13px;
            text-align: center;
            margin-top: 50px;
            color: #7D7D7D;
        }
    </style>
""", unsafe_allow_html=True)


# ===========================================================
# 4. SIDEBAR NAVIGATION
# ===========================================================
st.sidebar.title("📌 Navigation Menu")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "📝 Model Input", "📊 Prediction Page"]
)


# ===========================================================
# 5. HOME PAGE
# ===========================================================
if page == "🏠 Home":
    st.markdown('<p class="title">🔍 AML Narrative Classification System</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Automated Compliance Screening Using FinBERT</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    ### ⭐ Overview  
    This system analyzes transaction narratives and predicts whether they are **Compliant** or **Non-Compliant** using:

    - 🤖 **FinBERT Transformer Model**  
      Pre-trained on financial text for high-accuracy compliance interpretation.

    ### ⭐ Features
    - Real-time text classification  
    - Probability scoring  
    - Clean, user-friendly interface  
    - Deployed on Streamlit Cloud  
    """)

    st.markdown("---")
    st.markdown('<p class="footer">AML Automation System • Powered by FinBERT NLP</p>', unsafe_allow_html=True)


# ===========================================================
# 6. MODEL INPUT PAGE
# ===========================================================
elif page == "📝 Model Input":

    st.markdown('<p class="title">📝 Enter Narrative for Screening</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your text will be analyzed by the FinBERT model.</p>', unsafe_allow_html=True)
    st.markdown("---")

    user_text = st.text_area(
        "Enter Transaction Narrative:",
        height=200,
        placeholder="Example: Large transfer to Dubai with missing KYC documents..."
    )

    if st.button("Save Narrative"):
        if user_text.strip() != "":
            st.session_state["user_text"] = user_text
            st.success("Narrative saved! Go to 'Prediction Page' to classify it.")
        else:
            st.warning("Please enter text before saving.")

    st.markdown('<p class="footer">Input Page • AML System</p>', unsafe_allow_html=True)


# ===========================================================
# 7. PREDICTION PAGE
# ===========================================================
elif page == "📊 Prediction Page":

    st.markdown('<p class="title">📊 Prediction Results</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Classification Using the FinBERT Model</p>', unsafe_allow_html=True)
    st.markdown("---")

    if "user_text" not in st.session_state:
        st.warning("Please go to 'Model Input' and enter a narrative first.")
    else:
        text = st.session_state["user_text"]

        with st.spinner("Running FinBERT model..."):
            finbert = predict_finbert(text)

        st.markdown("### 🤖 FinBERT Model Prediction")
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.write("**Prediction:**", finbert["prediction"])
        st.write("**Non-Compliant Probability:**", round(finbert["prob_non_compliant"], 4))
        st.markdown('</div>', unsafe_allow_html=True)

        st.success("🎉 Classification complete!")

    st.markdown('<p class="footer">Prediction Page • AML System</p>', unsafe_allow_html=True)
