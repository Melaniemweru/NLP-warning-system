# ===============================================================
# FINBERT PREDICTION HELPER (src/models/finbert_predict.py)
# ===============================================================

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------
# Lazy loading (FinBERT loads only once)
# ---------------------------------------------------------------
_MODEL = None
_TOKENIZER = None

MODEL_NAME = "yiyanghkust/finbert-tone"   # Original FinBERT-tone


def load_finbert():
    """
    Loads the original FinBERT model ONCE (cached in memory).
    """
    global _MODEL, _TOKENIZER

    if _MODEL is None or _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _MODEL.eval()  # put into inference mode

    return _TOKENIZER, _MODEL



def predict_finbert(text: str, threshold: float = 0.30):
    """
    Predict AML risk using FinBERT sentiment model.

    FINBERT-TONE DEFAULT CLASS ORDER (DOCUMENTED):
        0 = neutral
        1 = positive
        2 = negative

    This model sometimes does NOT load id2label correctly,
    so we hardcode the mapping.
    """

    tokenizer, model = load_finbert()

    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    # ------------------------------------------------------
    # MANUAL FIX — Hardcoded FinBERT class mapping
    # ------------------------------------------------------
    id2label = {
        0: "neutral",
        1: "positive",
        2: "negative"
    }

    # FinBERT negative class is ALWAYS index 2
    neg_index = 2
    prob_negative = float(probs[neg_index])

    # Raw predicted sentiment (index of highest probability)
    raw_sentiment_label = id2label[int(probs.argmax())]

    # ------------------------------------------------------
    # AML DECISION LOGIC
    # ------------------------------------------------------
    prediction = "Non-Compliant" if prob_negative >= threshold else "Compliant"

    # Return everything for debugging
    return {
        "prediction": prediction,
        "prob_non_compliant": prob_negative,
        "raw_sentiment_label": raw_sentiment_label,
        "raw_probabilities": probs.tolist(),
        "id2label": id2label,            # added for debugging
        "neg_index": neg_index           # added for debugging
    }
