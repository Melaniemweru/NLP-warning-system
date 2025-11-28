import os
import joblib

# -----------------------------------------------------------
# PATHS TO MODEL ARTIFACTS
# -----------------------------------------------------------
THIS_DIR = os.path.dirname(__file__)
VECT_PATH = os.path.join(THIS_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(THIS_DIR, "logreg_tfidf.pkl")

# Cached objects
_VECTORIZER = None
_MODEL = None


# -----------------------------------------------------------
# LOAD VECTORIZER AND MODEL (only once)
# -----------------------------------------------------------
def _load_artifacts():
    global _VECTORIZER, _MODEL

    if _VECTORIZER is None:
        _VECTORIZER = joblib.load(VECT_PATH)

    if _MODEL is None:
        _MODEL = joblib.load(MODEL_PATH)

    return _VECTORIZER, _MODEL


# -----------------------------------------------------------
# PREDICT FUNCTION USED BY STREAMLIT
# -----------------------------------------------------------
def predict_narrative(text: str):
    """
    Predict whether narrative is Compliant or Non-Compliant.
    Returns:
        {
            "prediction": "Compliant" or "Non-Compliant",
            "prob_non_compliant": <float>
        }
    """
    vect, model = _load_artifacts()

    # Transform text
    X = vect.transform([text])

    # Probabilities
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)

    # Find probability of "Non-Compliant"
    try:
        idx_non = classes.index("Non-Compliant")
    except ValueError:
        idx_non = 1  # fallback in case classes are ["Compliant", "Non-Compliant"]

    prob_non = float(proba[idx_non])

    # Final prediction
    pred = model.predict(X)[0]

    return {
        "prediction": pred,
        "prob_non_compliant": prob_non
    }
