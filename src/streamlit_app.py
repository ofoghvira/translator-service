# streamlit_app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json


MODEL_PATH = "model/nllb-200-distilled-600M"


LANGUAGES = {
    "English": "eng_Latn",
    "Farsi": "pes_Arab",
    "German": "deu_Latn",
    "Arabic": "arb_Arab",
    "Turkish": "tur_Latn",
    "Spanish": "spa_Latn",
}

SRC_LANG = "eng_Latn"


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True)
    return tokenizer, model

tokenizer, model = load_model()

# ----------------------------- UI -----------------------------
st.title("🌐 Multilingual Text Translator")

st.markdown("Paste your input JSON array (title + text) below:")

example_input = '''[
  {
    "title": "description",
    "text": "this product is vira' product"
  },
  {
    "title": "information",
    "text": "this is information of product."
  }
]'''

input_json = st.text_area("Input Data", value=example_input, height=200)

target_langs = st.multiselect("Select Target Languages", options=list(LANGUAGES.keys()), default=["Farsi", "German"])

if st.button("Translate"):
    try:
        input_data = json.loads(input_json)
        output = []

        for item in input_data:
            translations = {}
            for lang_name in target_langs:
                lang_code = LANGUAGES[lang_name]

                tokenizer.src_lang = SRC_LANG
                encoded = tokenizer(item["text"], return_tensors="pt")
                generated = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id[lang_code])
                translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

                translations[lang_code] = translated

            output.append({
                "title": item["title"],
                "translate": translations
            })

        st.success("✅ Translations Completed")
        st.json(output)

        st.download_button("📥 Download Result as JSON", data=json.dumps(output, indent=2), file_name="translations.json", mime="application/json")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")