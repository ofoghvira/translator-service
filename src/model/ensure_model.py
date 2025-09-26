# src/model/ensure_model.py
# comments in English only

from pathlib import Path
from huggingface_hub import snapshot_download

HF_REPO_ID = "facebook/nllb-200-distilled-600M"
LOCAL_DIR = Path(__file__).resolve().parent / "nllb-200-distilled-600M"


def ensure_model() -> str:
    """
    Ensure that the model weights exist locally.
    If not found, download them once from HuggingFace Hub.
    Returns the local directory path as string.
    """
    if not LOCAL_DIR.exists() or not any(LOCAL_DIR.iterdir()):
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=str(LOCAL_DIR),
            local_dir_use_symlinks=False,
            # optionally restrict patterns:
            # allow_patterns=["*.json", "*.safetensors", "tokenizer*"]
        )
    return str(LOCAL_DIR)


if __name__ == "__main__":
    # for testing: run `python src/model/ensure_model.py`
    path = ensure_model()
    print(f"Model is ready at: {path}")
