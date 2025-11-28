import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------------------------------------------
# Lazy loading so FinBERT loads only once
# -------------------------------------------------------------------

_MODEL = None
_TOKENIZER = None

MODEL_NAME = "yiyanghkust/finbert-tone"   # Original FinBERT-tone


def load_finbert():
    """
    Loads the original FinBERT model and tokenizer only once.
    Uses the REAL FinBERT labels (negative, neutral, positive).
    """

    global _MODEL, _TOKENIZER

    if _MODEL is None or _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _MODEL.eval()  # important for inference

    return _TOKENIZER, _MODEL



def predict_finbert(text: str):
    """
    Predict AML risk using ORIGINAL FinBERT sentiment model.
    Maps:
        negative → Non-Compliant (high AML risk)
        neutral/positive → Compliant
    
    Returns:
        - prediction: "Compliant" or "Non-Compliant"
        - prob_non_compliant: probability of being Non-Compliant
        - raw_sentiment_label: FinBERT sentiment label (negative/neutral/positive)
        - raw_probabilities: list of probabilities for all 3 classes
    """

    tokenizer, model = load_finbert()

    # Tokenize input
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

    # Real FinBERT labels
    id2label = model.config.id2label       # {0:'negative',1:'neutral',2:'positive'}

    # Identify negative class index
    neg_index = None
    for idx, lab in id2label.items():
        if str(lab).lower().startswith("negative"):
            neg_index = int(idx)
            break

    # Fallback if not found (rare)
    if neg_index is None:
        neg_index = 0

    prob_negative = float(probs[neg_index])  
    raw_sentiment = id2label[int(probs.argmax())]

    # AML decision rule  
    # You can adjust threshold (0.40–0.50 recommended)
    prediction = "Non-Compliant" if prob_negative >= 0.40 else "Compliant"

    return {
        "prediction": prediction,
        "prob_non_compliant": prob_negative,
        "raw_sentiment_label": raw_sentiment,
        "raw_probabilities": probs.tolist()
    }
