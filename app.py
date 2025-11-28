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
        # DEBUG: Check which module is being used
        # --------------------------------------------------------
        st.write("MODULE CALLED:", predict_finbert.__module__)

        st.markdown("### 🤖 FinBERT Model Prediction")
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.write("**Prediction:**", finbert_output["prediction"])
        st.write("**Non-Compliant Probability:**", round(finbert_output["prob_non_compliant"], 4))
        st.write("**Sentiment Detected:**", finbert_output["raw_sentiment_label"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.success("🎉 Classification complete!")

        # --------------------------------------------------------
        # DEBUG: Show raw model outputs
        # --------------------------------------------------------
        with st.expander("🔎 Debug: Raw Model Outputs"):
            st.write("id2label:", finbert_output.get("id2label"))
            st.write("Negative class index:", finbert_output.get("neg_index"))
            st.write("Raw probabilities:", finbert_output.get("raw_probabilities"))
            st.write("Raw sentiment label:", finbert_output.get("raw_sentiment_label"))

    st.markdown('<p class="footer">Prediction Page • AML System</p>', unsafe_allow_html=True)
