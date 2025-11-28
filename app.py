# ===========================================================
# STREAMLIT APP (app.py)
# ===========================================================

import streamlit as st
from src.models.finbert_predict import predict_finbert   # <-- IMPORTANT FIX

# ===========================================================
# STREAMLIT PAGE CONFIG
# ===========================================================
st.set_page_config(
    page_title="AML Narrative Classification System",
    layout="wide",
    page_icon="🔍"
)

# ===========================================================
# CUSTOM STYLING
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
        .footer {
            font-size: 13px;
            text-align: center;
            margin-top: 50px;
            color: #7D7D7D;
        }
    </style>
""", unsafe_allow_html=True)

# ===========================================================
# SIDEBAR NAVIGATION
# ===========================================================
st.sidebar.title("📌 Navigation Menu")

page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "📝 Model Input", "📊 Prediction Page"]
)

# Let user choose threshold for AML sensitivity
threshold = st.sidebar.slider(
    "Non-Compliant Threshold",
    min_value=0.10,
    max_value=0.60,
    value=0.30,
    step=0.05,
    help="Lower = more sensitive (more Non-Compliant flags)"
)

# ===========================================================
# 1. HOME PAGE
# ===========================================================
if page == "🏠 Home":
    st.markdown('<p class="title">🔍 AML Narrative Classification System</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Automated Compliance Screening Using FinBERT</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    ### ⭐ Overview  
    This system analyzes transaction narratives and predicts whether they are **Compliant** or **Non-Compliant** using:

    - 🤖 **FinBERT Transformer Model**
      Pre-trained on financial text.

    ### ⭐ Features
    - Real-time classification
    - Probability scoring
    - Simple interface
    - Streamlit deployment
    """)

    st.markdown("---")
    st.markdown('<p class="footer">AML Automation System • Powered by FinBERT NLP</p>', unsafe_allow_html=True)

# ===========================================================
# 2. MODEL INPUT PAGE
# ===========================================================
elif page == "📝 Model Input":

    st.markdown('<p class="title">📝 Enter Narrative for Screening</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your text will be analyzed by FinBERT.</p>', unsafe_allow_html=True)
    st.markdown("---")

    user_text = st.text_area(
        "Enter Transaction Narrative:",
        height=200,
        placeholder="Example: Transfer of 500,000 USD to foreign account with unexplained purpose..."
    )

    if st.button("Save Narrative"):
        if user_text.strip() != "":
            st.session_state["user_text"] = user_text
            st.success("Narrative saved successfully! Go to 'Prediction Page'.")
        else:
            st.warning("Please enter a narrative before saving.")

    st.markdown('<p class="footer">Input Page • AML System</p>', unsafe_allow_html=True)

# ===========================================================
# 3. PREDICTION PAGE
# ===========================================================
elif page == "📊 Prediction Page":

    st.markdown('<p class="title">📊 Prediction Results</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Classification using FinBERT</p>', unsafe_allow_html=True)
    st.markdown("---")

    if "user_text" not in st.session_state:
        st.warning("Please enter a narrative first on the Model Input page.")
    else:
        text = st.session_state["user_text"]

        with st.spinner("Running FinBERT model..."):
            finbert_output = predict_finbert(text, threshold=threshold)

        # --------------------------------------------------------
        # DEBUG: Which module is actually being used?
        # --------------------------------------------------------
        st.write("MODULE CALLED:", predict_finbert.__module__)

        # --------------------------------------------------------
        # MAIN OUTPUT
        # --------------------------------------------------------
        st.markdown("### 🤖 FinBERT Model Prediction")
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.write("**Prediction:**", finbert_output["prediction"])
        st.write("**Non-Compliant Probability:**", round(finbert_output["prob_non_compliant"], 4))
        st.write("**Sentiment Detected:**", finbert_output["raw_sentiment_label"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.success("🎉 Classification complete!")

        # --------------------------------------------------------
        # FULL DEBUG INFORMATION
        # --------------------------------------------------------
        with st.expander("🔎 Debug: Raw Model Outputs"):
            st.write("id2label:", finbert_output.get("id2label"))
            st.write("Negative class index:", finbert_output.get("neg_index"))
            st.write("Raw probabilities:", finbert_output.get("raw_probabilities"))
            st.write("Raw sentiment label:", finbert_output.get("raw_sentiment_label"))

    st.markdown('<p class="footer">Prediction Page • AML System</p>', unsafe_allow_html=True)
