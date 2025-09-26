
# AI Translator using NLLB-200

This project provides a web-based interface using **Streamlit** to
translate text or JSON documents into multiple languages using Meta's
**NLLB-200** model. The app supports translation into English, Persian,
Arabic, Turkish, Spanish, and more.

## Features

-   Upload **text** or **JSON** files for translation\
-   Translate into multiple target languages simultaneously\
-   Display original vs translated text in a clean interface\
-   Provide both **web app** (Streamlit) and **API service**


### What's inside?

-   **Architecture**: Meta AI's **NLLB-200 (No Language Left Behind)**
    distilled model (`nllb-200-distilled-600M`)
-   **Task & Labels**: Machine Translation --- multilingual text-to-text
    translation
-   **Model Files**: Core artifacts such as tokenizer, config, and
    weights are provided on Hugging Face (see **Files and versions**)
-   **License**: CC-BY-NC 4.0
-   **Supported Languages**: 200+ (English, Persian, Arabic, Turkish,
    Spanish, ...)

## How to Run

1.  **Install the required libraries**:

Install all dependencies using the `requirements.txt` file:

``` bash
pip install -r requirements.txt
```

[requirements.txt](./requirements.txt)

## ⚠️ Important Note about Model Download

If the model does not load for you or you encounter any model-related errors,
make sure to run the helper script [`ensure_model.py`](./src/model/ensure_model.py).
This script will automatically download the required model
(**facebook/nllb-200-distilled-600M**) from Hugging Face Hub.

- The model will be cached locally inside
  `src/model/nllb-200-distilled-600M/`.
- You do **not** need to manually place the weights in the repository.
- Make sure you have internet access during the first run; subsequent
  runs will use the cached copy.

2.  **Run the Streamlit app**:

``` bash
streamlit run streamlit_app.py
```

[streamlit_app.py](src/streamlit_app.py)

This will launch the web application in your browser.

3.  **Run the API**:

``` bash
python api.py
```

[api.py](src/api.py)

The API will be available at http://localhost:8000 (or your configured
port).
