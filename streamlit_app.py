# streamlit_app.py
import io
import re
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pydub
import streamlit as st
from streamlit_mic_recorder import mic_recorder  # click to start, click to stop [web:89]

from transformers import WhisperForConditionalGeneration, WhisperProcessor

from om_transliterator import Transliterator  # Kannada -> Latin transliteration [web:218]


# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="speak2list_kannada", layout="centered")
st.title("speak2list_kannada")

MODEL_ID = "vasista22/whisper-kannada-medium"


# ----------------------------
# Cached resources
# ----------------------------
@st.cache_resource
def load_asr(model_id: str):
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    return processor, model


@st.cache_resource
def load_transliterator():
    return Transliterator()


processor, model = load_asr(MODEL_ID)
transliterator = load_transliterator()


# ----------------------------
# Session state init (avoid key name 'items')
# ----------------------------
if "audio_seg" not in st.session_state:
    st.session_state.audio_seg = pydub.AudioSegment.silent(duration=0)

if "transcript_kn" not in st.session_state:
    st.session_state.transcript_kn = ""

if "shopping_items" not in st.session_state:
    st.session_state.shopping_items = []

if "list_text" not in st.session_state:
    st.session_state.list_text = ""


# ----------------------------
# Helpers
# ----------------------------
def reset_all():
    st.session_state.audio_seg = pydub.AudioSegment.silent(duration=0)
    st.session_state.transcript_kn = ""
    st.session_state.shopping_items = []
    st.session_state.list_text = ""


def audio_dict_to_segment(audio: dict) -> pydub.AudioSegment:
    # mic_recorder returns None or a dict containing recorded audio bytes [web:89]
    b = audio["bytes"]
    seg = pydub.AudioSegment.from_file(io.BytesIO(b))
    return seg.set_channels(1)


def keep_speech_only(
    seg: pydub.AudioSegment,
    min_silence_len_ms: int = 500,
    silence_thresh_dbfs: int = -40,
    keep_silence_ms: int = 150,
) -> pydub.AudioSegment:
    chunks = pydub.silence.split_on_silence(
        seg,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_thresh_dbfs,
        keep_silence=keep_silence_ms,
    )
    if not chunks:
        return pydub.AudioSegment.silent(duration=0)
    out = pydub.AudioSegment.silent(duration=0)
    for c in chunks:
        out += c
    return out


def transcribe_kn(seg: pydub.AudioSegment) -> str:
    if len(seg) < 300:
        return ""

    seg = seg.set_frame_rate(16000).set_channels(1)
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    denom = np.max(np.abs(samples)) or 1.0
    samples = samples / denom

    inputs = processor(samples, sampling_rate=16000, return_tensors="pt")
    predicted_ids = model.generate(inputs.input_features)
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text.strip()


def extract_items(transcript: str) -> List[Tuple[str, Optional[str]]]:
    # Baseline: split on commas/newlines, parse an optional leading numeric quantity token.
    parts = re.split(r"[,\n]+", transcript)
    out: List[Tuple[str, Optional[str]]] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue

        m = re.match(
            r"^(\d+(\.\d+)?\s*(kg|g|ಲೀ|ml|l|pcs|piece|pieces)?)\s+(.*)$",
            p,
            flags=re.IGNORECASE,
        )
        if m:
            qty = m.group(1).strip()
            item = m.group(4).strip()
            out.append((item, qty))
        else:
            out.append((p, None))
    return out


def format_list(items: List[Tuple[str, Optional[str]]], out_lang: str) -> str:
    lines = []
    for item, qty in items:
        item_out = item
        if out_lang == "English (transliteration)":
            item_out = transliterator.knda_to_latn(item_out)  # basic usage [web:218]

        lines.append(f"{item_out} - {qty}" if qty else f"{item_out}")

    return "\n".join(lines).strip()


def build_txt_bytes(out_lang: str) -> bytes:
    speech_only = keep_speech_only(st.session_state.audio_seg)
    transcript_kn = transcribe_kn(speech_only)
    shopping_items = extract_items(transcript_kn)
    list_text = format_list(shopping_items, out_lang)

    # Store for preview on next rerun (don’t rely on UI inside download callable) [web:78]
    st.session_state.transcript_kn = transcript_kn
    st.session_state.shopping_items = shopping_items
    st.session_state.list_text = list_text

    return (list_text + "\n").encode("utf-8")


# ----------------------------
# Sidebar: language + actions
# ----------------------------
with st.sidebar:
    st.header("Settings")
    out_lang = st.selectbox(
        "Download language",
        ["Kannada", "English (transliteration)"],
        index=0,
    )
    st.caption("English here means transliteration (Kannada → Latin), not translation.")

    if st.button("Clear / start fresh"):
        reset_all()


# ----------------------------
# Recorder
# ----------------------------
st.subheader("Record")
audio = mic_recorder(
    start_prompt="Start recording",
    stop_prompt="Stop recording",
    just_once=True,               # return audio once after stop [web:89]
    use_container_width=True,
    key="recorder",
)

if audio is not None:
    seg = audio_dict_to_segment(audio)
    st.session_state.audio_seg += seg
    st.success(
        f"Added recording: ~{len(seg)/1000:.1f}s. "
        f"Total: ~{len(st.session_state.audio_seg)/1000:.1f}s"
    )


# ----------------------------
# Single button: generate + download
# ----------------------------
st.subheader("Generate & download")

date_tag = datetime.now().strftime("%Y-%m-%d")
fname = f"speak2list_kannada_{date_tag}.txt"

# Use a callable so ASR runs only when clicked. Streamlit notes that
# Streamlit commands inside the callable are ignored. [web:78]
st.download_button(
    label="Generate & Download",
    data=lambda: build_txt_bytes(out_lang),  # generate only for selected language [web:78]
    file_name=fname,
    mime="text/plain",
    key="dl",
)


# ----------------------------
# Preview
# ----------------------------
st.subheader("Preview")
st.text_area("Transcript (Kannada)", value=st.session_state.transcript_kn, height=120)
st.code(st.session_state.list_text or "(No items yet)", language="text")
