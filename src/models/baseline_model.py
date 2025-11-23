import os
import joblib

THIS_DIR = os.path.dirname(__file__)
VECT_PATH = os.path.join(THIS_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(THIS_DIR, "logreg_tfidf.pkl")

_VECTORIZER = None
_MODEL = None

def _load_artifacts():
    global _VECTORIZER, _MODEL
    if _VECTORIZER is None:
        _VECTORIZER = joblib.load(VECT_PATH)
    if _MODEL is None:
        _MODEL = joblib.load(MODEL_PATH)
    return _VECTORIZER, _MODEL

def predict_narrative(text: str):
    vect, model = _load_artifacts()
    X = vect.transform([text])
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]

    return {
        "prediction": pred,
        "probabilities": proba.tolist(),
        "classes": model.classes_.tolist()
    }
