# speak2list_kannada

speak2list_kannada is a Streamlit web app (Android Chrome friendly) that lets users record Kannada speech, transcribes it into a shopping list, and downloads the result as `form.txt`. It supports downloading the list in **Kannada script** or **English transliteration** (Kannada → Latin letters).

## Features
- Tap **Start recording** → speak items and quantities → tap **Stop recording**
- Record multiple clips (they get appended)
- Choose download language: Kannada or English (transliteration)
- Click **Download list (generate form.txt)** to transcribe + build the list (only for the selected language)
- **Clear / start fresh** button to remove all recordings and outputs
- Download generated output via a Streamlit download button (`form.txt`)

## Tech stack
- UI: Streamlit
- Audio recording: `streamlit-mic-recorder`
- ASR: `vasista22/whisper-kannada-medium` (Whisper fine-tune)
- Audio cleanup: silence trimming (pydub)
- List parsing: basic item/quantity parsing (regex baseline)
- Transliteration: `om-transliterator` (Kannada → Latin)

## Project structure
```text
.
├── app.py
├── requirements.txt
└── README.md
