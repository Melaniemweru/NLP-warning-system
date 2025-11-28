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
    Loads the original FinBERT model ONCE (cached in memory)
    """
    global _MODEL, _TOKENIZER

    if _MODEL is None or _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _MODEL.eval()

    return _TOKENIZER, _MODEL



def predict_finbert(text: str, threshold: float = 0.30):
    """
    Predict AML risk using ORIGINAL FinBERT sentiment model.

    Mapping rule:
        negative → Non-Compliant (higher AML risk)
        neutral/positive → Compliant

    threshold controls sensitivity (recommended: 0.25–0.40)
    """

    tokenizer, model = load_finbert()

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # Forward Pass
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    # ------------------------------
    # Identify FinBERT's negative class index
    # ------------------------------
    id2label = model.config.id2label

    neg_index = None
    for idx, lab in id2label.items():
        if "neg" in str(lab).lower():
            neg_index = int(idx)
            break

    if neg_index is None:
        neg_index = 0  # fallback

    prob_negative = float(probs[neg_index])

    # Get raw sentiment
    raw_sentiment_label = id2label[int(probs.argmax())]

    # ------------------------------
    # AML DECISION LOGIC
    # ------------------------------
    prediction = "Non-Compliant" if prob_negative >= threshold else "Compliant"

    return {
        "prediction": prediction,
        "prob_non_compliant": prob_negative,
        "raw_sentiment_label": raw_sentiment_label,
        "raw_probabilities": probs.tolist()
    }
