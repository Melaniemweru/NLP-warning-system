import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------------------------------------------
# Lazy loading: FinBERT loads only once when called the first time
# -------------------------------------------------------------------

_MODEL = None
_TOKENIZER = None

MODEL_NAME = "yiyanghkust/finbert-tone"

def load_finbert():
    """
    Loads FinBERT model and tokenizer only once.
    Returns loaded tokenizer and model.
    """

    global _MODEL, _TOKENIZER

    if _MODEL is None or _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            id2label={0: "Compliant", 1: "Non-Compliant"},
            label2id={"Compliant": 0, "Non-Compliant": 1},
            ignore_mismatched_sizes=True
        )
        _MODEL.eval()   # important for inference

    return _TOKENIZER, _MODEL


def predict_finbert(text: str):
    """
    Uses the fine-tuned FinBERT to classify a transaction narrative.
    Returns:
        - predicted label ("Compliant" / "Non-Compliant")
        - probability of Non-Compliant
    """

    tokenizer, model = load_finbert()

    # Tokenize text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    # Disable gradients for faster inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Predicted class
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_id = int(probs.argmax())

    label = model.config.id2label[pred_id]
    prob_non_compliant = float(probs[1])  # class 1 = Non-Compliant

    return {
        "prediction": label,
        "prob_non_compliant": prob_non_compliant,
    }
