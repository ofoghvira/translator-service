# api.py (Flask version)
from typing import List, Dict, Optional
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -------- Config --------
MODEL_PATH = "model/nllb-200-distilled-600M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS = 512

# Map human-friendly language codes to NLLB codes
# Note: "ge" is mapped to German to match the sample PDF.
NLLB_LANG: Dict[str, str] = {
    "fa": "pes_Arab",   # or "fas_Arab" depending on your tokenizer
    "ar": "arb_Arab",
    "tr": "tur_Latn",
    "es": "spa_Latn",
    "en": "eng_Latn",
    "de": "deu_Latn",
    "ge": "deu_Latn",
    "fr": "fra_Latn",
    "ru": "rus_Cyrl",
    "it": "ita_Latn",
    "pt": "por_Latn",
}

SRC_LANG_CANDIDATES = ["fas_Arab", "pes_Arab"]

# -------- Init --------
app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE)

# Choose Persian source language token if available
src_lang: Optional[str] = None
if hasattr(tokenizer, "lang_code_to_id"):
    for cand in SRC_LANG_CANDIDATES:
        if cand in tokenizer.lang_code_to_id:
            src_lang = cand
            break
if src_lang is None:
    src_lang = SRC_LANG_CANDIDATES[0]

try:
    tokenizer.src_lang = src_lang
except Exception:
    pass

# -------- Helpers --------
def _lang_to_nllb(lang_code: str) -> Optional[str]:
    lang_code = (lang_code or "").lower().strip()
    return NLLB_LANG.get(lang_code)

def _translate_text(txt: str, tgt_lang_code: str) -> str:
    tgt = _lang_to_nllb(tgt_lang_code)
    if not tgt:
        raise ValueError(f"Unsupported language code: {tgt_lang_code}")

    inputs = tokenizer(
        txt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS
    ).to(DEVICE)

    forced_bos_token_id = None
    if hasattr(tokenizer, "lang_code_to_id"):
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt)

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
            forced_bos_token_id=forced_bos_token_id
        )
    out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    return out

# -------- Routes --------
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "device": DEVICE, "model": str(MODEL_PATH)})

@app.route("/translate", methods=["POST"])
def translate():
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception as e:
        return jsonify({"detail": f"Invalid JSON: {e}"}), 400

    if not isinstance(payload, dict):
        return jsonify({"detail": "Request body must be a JSON object"}), 400

    data = payload.get("data")
    languages = payload.get("languages")

    if not isinstance(data, list) or not data:
        return jsonify({"detail": "Field 'data' must be a non-empty array"}), 400
    if not isinstance(languages, list) or not languages:
        return jsonify({"detail": "Field 'languages' must be a non-empty array"}), 400

    results: List[Dict] = []
    try:
        for item in data:
            title = item.get("title")
            text = item.get("text")
            if not isinstance(title, str) or not isinstance(text, str):
                return jsonify({"detail": "Each item in 'data' must have string fields 'title' and 'text'"}), 400

            translations: Dict[str, str] = {}
            for lang in languages:
                if not isinstance(lang, str):
                    return jsonify({"detail": "Languages must be an array of strings"}), 400
                try:
                    translations[lang] = _translate_text(text, lang)
                except ValueError as ve:
                    return jsonify({"detail": str(ve)}), 400
                except Exception as ex:
                    return jsonify({"detail": f"Translation failed for '{lang}': {ex}"}), 500

            results.append({"title": title, "translate": translations})

        return jsonify({"afterProcessData": results}), 200

    except Exception as e:
        return jsonify({"detail": f"Unexpected error: {e}"}), 500

if __name__ == "__main__":
    # For production, use a WSGI server like gunicorn:
    # gunicorn -w 2 -b 0.0.0.0:8000 api:app
    app.run(host="0.0.0.0", port=8000, debug=False)
