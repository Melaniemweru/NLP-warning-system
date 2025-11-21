import streamlit as st
from src.models.baseline_model import predict_narrative
from src.models.finbert_predict import predict_finbert

st.set_page_config(page_title="AML Warning System", layout="wide")

st.title("🔍 AML Narrative Classification System")
st.write(
    "Enter a transaction narrative below to classify it as "
    "**Compliant** or **Non-Compliant**."
)

text_input = st.text_area("Transaction Narrative", height=200)

if st.button("Classify"):
    if not text_input.strip():
        st.warning("Please enter text.")
    else:
        # Run both models
        baseline = predict_narrative(text_input)
        finbert = predict_finbert(text_input)

        st.subheader("📌 Baseline Model (TF-IDF + Logistic Regression)")
        st.write("Prediction:", baseline["prediction"])
        st.write(
            "Probability of Non-Compliant:",
            round(baseline["prob_non_compliant"], 4),
        )

        st.markdown("---")

        st.subheader("📌 FinBERT Model (Fine-Tuned)")
        st.write("Prediction:", finbert["prediction"])
        st.write(
            "Probability of Non-Compliant:",
            round(finbert["prob_non_compliant"], 4),
        )

        st.markdown("---")
        st.success("Classification complete.")
