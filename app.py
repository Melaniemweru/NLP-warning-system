import streamlit as st
from src.models.baseline_model import predict_narrative
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load original FinBERT
@st.cache_resource
def load_finbert():
    MODEL_NAME = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_finbert()

def predict_finbert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
    compliant_prob = probs[0]
    non_compliant_prob = probs[1]

    label = "Non-Compliant" if non_compliant_prob > 0.5 else "Compliant"

    return {
        "prediction": label,
        "prob_non_compliant": float(non_compliant_prob)
    }

# ---------------- UI -------------------

st.set_page_config(page_title="AML Warning System", layout="wide")

st.title("🔍 AML Narrative Classification System")
st.write(
    "Enter a transaction narrative below to classify it as "
    "**Compliant** or **Non-Compliant** using TF-IDF and FinBERT."
)

text_input = st.text_area("Transaction Narrative", height=200)

if st.button("Classify"):
    if not text_input.strip():
        st.warning("Please enter text.")
    else:
        baseline = predict_narrative(text_input)
        finbert = predict_finbert(text_input)

        st.subheader("📌 Baseline Model (TF-IDF + Logistic Regression)")
        st.write("Prediction:", baseline["prediction"])
        st.write(
            "Probability of Non-Compliant:",
            round(baseline["prob_non_compliant"], 4),
        )

        st.markdown("---")

        st.subheader("📌 FinBERT Model (Original FinBERT – Non-Fine-Tuned)")
        st.write("Prediction:", finbert["prediction"])
        st.write(
            "Probability of Non-Compliant:",
            round(finbert["prob_non_compliant"], 4),
        )

        st.markdown("---")
        st.success("Classification complete.")
