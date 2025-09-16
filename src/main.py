from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

MODEL_PATH = "model/nllb-200-distilled-600M"


SRC_LANG = "pes_Arab"


TARGET_LANGS = {
    "en": "eng_Latn",
    "ar": "arb_Arab",
    "tr": "tur_Latn",
    "es": "spa_Latn",
}



text = input("Enter your text: ").strip()


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True, use_safetensors=True)


tokenizer.src_lang = SRC_LANG


inputs = tokenizer(text, return_tensors="pt")


translations = {}
for iso, tgt_code in TARGET_LANGS.items():
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_code)
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos,
        max_new_tokens=160,
    )
    translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    translations[iso] = translated


result = {"input": text, "translations": translations}
print(json.dumps(result, ensure_ascii=False, indent=2))
