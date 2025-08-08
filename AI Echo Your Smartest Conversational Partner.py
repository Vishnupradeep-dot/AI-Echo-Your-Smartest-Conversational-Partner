

import os, glob, re, json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.set_page_config(page_title="AI Echo: Sentiment Inference", layout="centered")
st.title("🧠 AI Echo: Sentiment Inference (Pretrained Model)")
st.caption("Fast predictions using TF-IDF (1–2 grams) + Logistic Regression pipeline.")

# --- Debug helpers: show where the app is running and which PKLs it can see
st.write("**Working directory:**", os.getcwd())
st.write("**PKL files visible here:**", [os.path.basename(p) for p in glob.glob("*.pkl")])

# --- Sidebar: allow overriding the model path if needed
MODEL_PATH = st.sidebar.text_input("Model path (.pkl)", value="sentiment_pipeline.pkl")
VOCAB_PATH = "tfidf_vocabulary.pkl"   # optional
META_PATH  = "model_meta.json"        # optional

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

def try_load_vocab_and_meta(vocab_path: str, meta_path: str):
    vocab, meta = None, None
    if Path(vocab_path).exists():
        try:
            vocab = joblib.load(vocab_path)
        except Exception as e:
            st.warning(f"Could not load vocabulary file ({vocab_path}): {e}")
    if Path(meta_path).exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            st.warning(f"Could not load metadata file ({meta_path}): {e}")
    return vocab, meta

def validate_against_pipeline(pipe, vocab, meta):
    msgs = []
    try:
        tfidf = pipe.named_steps["tfidf"]
        pvocab = tfidf.vocabulary_
        if vocab is not None and len(vocab) != len(pvocab):
            msgs.append(f"Vocabulary size mismatch: saved={len(vocab)}, in-model={len(pvocab)}")
        if meta is not None:
            vr = tuple(meta.get("vectorizer", {}).get("ngram_range", [])) or None
            mf = meta.get("vectorizer", {}).get("max_features", None)
            if vr and getattr(tfidf, "ngram_range", None) != vr:
                msgs.append(f"ngram_range mismatch: meta={vr}, in-model={getattr(tfidf, 'ngram_range', None)}")
            if mf and getattr(tfidf, "max_features", None) != mf:
                msgs.append(f"max_features mismatch: meta={mf}, in-model={getattr(tfidf, 'max_features', None)}")
            mclasses = meta.get("classes")
            if mclasses and list(getattr(pipe, "classes_", [])) != mclasses:
                msgs.append("Class label order differs from metadata.")
    except Exception as e:
        msgs.append(f"Validation check failed: {e}")
    return msgs

def clean_text_simple(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)

# ---------- Load model ----------
pipe = None
if Path(MODEL_PATH).exists():
    try:
        pipe = load_model(MODEL_PATH)
        st.success(f"Loaded model: {MODEL_PATH}")
    except Exception as e:
        st.error(f"Error loading model at '{MODEL_PATH}': {e}")
else:
    st.error(f"Model not found at '{MODEL_PATH}'. Upload a .pkl below, or fix the path in the sidebar.")

if pipe is None:
    uploaded = st.file_uploader("Upload a saved .pkl model", type=["pkl"], key="model_upload")
    if uploaded:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        tmp.write(uploaded.getbuffer())
        tmp.flush()
        pipe = load_model(tmp.name)
        st.success("Loaded uploaded model.")
    else:
        st.stop()

# Optional: validate against saved vocab & meta if present
vocab, meta = try_load_vocab_and_meta(VOCAB_PATH, META_PATH)
for msg in validate_against_pipeline(pipe, vocab, meta):
    st.warning(msg)

# ---------- Single text inference ----------
st.subheader("🔤 Single Review Prediction")
user_text = st.text_area("Enter a review:")
if st.button("Predict"):
    if user_text.strip():
        cleaned = clean_text_simple(user_text)
        pred = pipe.predict([cleaned])[0]
        st.write(f"**Predicted Sentiment:** {pred}")
        try:
            proba = pipe.predict_proba([cleaned])[0]
            labels = pipe.classes_
            st.write("Probabilities:")
            for label, p in zip(labels, proba):
                st.write(f"- {label}: {p:.2f}")
        except Exception:
            pass
    else:
        st.warning("Please enter some text.")

# ---------- Batch inference ----------
st.subheader("📦 Batch Prediction")
batch = st.file_uploader("Upload CSV/XLSX with a 'review' column", type=["csv", "xlsx"], key="batch")
if batch is not None:
    if batch.name.lower().endswith(".csv"):
        df_in = pd.read_csv(batch)
    else:
        df_in = pd.read_excel(batch)

    if "review" not in df_in.columns:
        st.error("File must contain a 'review' column.")
    else:
        df_out = df_in.copy()
        df_out["cleaned_review"] = df_out["review"].astype(str).apply(clean_text_simple)
        df_out["predicted_sentiment"] = pipe.predict(df_out["cleaned_review"])
        st.dataframe(df_out.head())
        st.download_button(
            "Download predictions (CSV)",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )
