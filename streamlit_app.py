import io
import re
import wave
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
from om_transliterator import Transliterator  # Kannada -> Latin transliteration [web:218]
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from speak2list_mic import speak2list_mic

st.set_page_config(page_title="speak2list_kannada", layout="centered")
st.title("speak2list_kannada")
st.caption("Android Chrome • Record Kannada voice • Preview • Generate & download list")

MODEL_ID = "vasista22/whisper-kannada-medium"

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
# Session state
# ----------------------------
if "wav_clips_16k" not in st.session_state:
    st.session_state.wav_clips_16k: List[np.ndarray] = []

if "transcript_kn" not in st.session_state:
    st.session_state.transcript_kn = ""

if "shopping_items" not in st.session_state:
    st.session_state.shopping_items: List[Tuple[str, Optional[str]]] = []

if "list_text" not in st.session_state:
    st.session_state.list_text = ""

if "last_lang" not in st.session_state:
    st.session_state.last_lang = "Kannada"

def reset_all():
    st.session_state.wav_clips_16k = []
    st.session_state.transcript_kn = ""
    st.session_state.shopping_items = []
    st.session_state.list_text = ""
    st.session_state.last_lang = "Kannada"

# ----------------------------
# Audio helpers
# ----------------------------
def read_pcm16_mono_from_wav_bytes(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    # Python's wave module reads PCM WAV. [web:269]
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        nchan = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)

    if sampwidth != 2:
        raise ValueError(f"Expected PCM16 WAV (sampwidth=2), got sampwidth={sampwidth}")
    x = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0

    if nchan > 1:
        x = x.reshape(-1, nchan).mean(axis=1).astype(np.float32)

    return x, int(sr)

def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return x.astype(np.float32)
    if len(x) < 2:
        return x.astype(np.float32)
    t_old = np.linspace(0, len(x) / sr, num=len(x), endpoint=False)
    n_new = int(len(x) * 16000 / sr)
    t_new = np.linspace(0, len(x) / sr, num=max(n_new, 2), endpoint=False)
    return np.interp(t_new, t_old, x).astype(np.float32)

def simple_vad_keep_speech(x: np.ndarray, sr: int = 16000) -> np.ndarray:
    if len(x) == 0:
        return x
    frame_ms, hop_ms = 30, 10
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if len(x) < frame or frame <= 0 or hop <= 0:
        return x

    eps = 1e-8
    rms = np.array(
        [np.sqrt(np.mean(x[i:i+frame] ** 2) + eps) for i in range(0, len(x) - frame + 1, hop)],
        dtype=np.float32,
    )
    if rms.size == 0:
        return x

    thr = float(np.percentile(rms, 20)) * 2.5
    speech = rms > max(thr, 0.01)
    if not speech.any():
        return x

    k = 5
    expanded = speech.copy()
    for i, v in enumerate(speech):
        if v:
            expanded[max(0, i-k):min(len(expanded), i+k+1)] = True

    intervals = []
    for idx, keep in enumerate(expanded):
        if keep:
            s = idx * hop
            e = min(s + frame, len(x))
            intervals.append((s, e))
    intervals.sort()

    merged = []
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))

    return np.concatenate([x[s:e] for s, e in merged]).astype(np.float32)

# ----------------------------
# NLP helpers
# ----------------------------
def transcribe_kn(wav16: np.ndarray) -> str:
    if wav16 is None or len(wav16) < 1600:
        return ""
    inputs = processor(wav16, sampling_rate=16000, return_tensors="pt")
    predicted_ids = model.generate(inputs.input_features)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

def extract_items(transcript: str) -> List[Tuple[str, Optional[str]]]:
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
            item_out = transliterator.knda_to_latn(item_out)  # Kannada -> Latin [web:218]
        lines.append(f"{item_out} - {qty}" if qty else f"{item_out}")
    return "\n".join(lines).strip()

def build_txt_bytes(out_lang: str) -> bytes:
    if not st.session_state.wav_clips_16k:
        st.session_state.transcript_kn = ""
        st.session_state.shopping_items = []
        st.session_state.list_text = ""
        st.session_state.last_lang = out_lang
        return b""

    wav16 = np.concatenate(st.session_state.wav_clips_16k).astype(np.float32)
    wav16_speech = simple_vad_keep_speech(wav16, sr=16000)

    transcript_kn = transcribe_kn(wav16_speech)
    shopping_items = extract_items(transcript_kn)
    list_text = format_list(shopping_items, out_lang)

    st.session_state.transcript_kn = transcript_kn
    st.session_state.shopping_items = shopping_items
    st.session_state.list_text = list_text
    st.session_state.last_lang = out_lang

    return (list_text + "\n").encode("utf-8")

# ----------------------------
# UI
# ----------------------------
c1, c2 = st.columns([1.2, 1])
with c1:
    out_lang = st.selectbox(
        "Download language",
        ["Kannada", "English (transliteration)"],
        index=0 if st.session_state.last_lang == "Kannada" else 1,
    )
with c2:
    st.write("")
    if st.button("Clear / start fresh", use_container_width=True):
        reset_all()

st.subheader("Record")
payload = speak2list_mic(key="mic")
st.write("mic payload:", payload)

if payload and payload.get("status") == "error":
    st.error(payload.get("error", "Microphone error"))
elif payload and payload.get("status") == "stopped" and payload.get("wav_bytes"):
    wav_bytes = bytes(payload["wav_bytes"])
    try:
        x, sr = read_pcm16_mono_from_wav_bytes(wav_bytes)
        x16 = resample_to_16k(x, sr)
        st.session_state.wav_clips_16k.append(x16)
        total_s = sum(len(c) for c in st.session_state.wav_clips_16k) / 16000.0
        st.success(f"Clip added. Total recorded: ~{total_s:.1f}s")
    except Exception as e:
        st.error(f"Could not decode WAV: {e}")

st.divider()

st.subheader("Preview")
st.text_area("Transcript (Kannada)", value=st.session_state.transcript_kn, height=120)
st.code(st.session_state.list_text or "(No items yet)", language="text")

st.subheader("Generate & download")
date_tag = datetime.now().strftime("%Y-%m-%d")
fname = f"speak2list_kannada_{date_tag}.txt"

st.download_button(
    label="Generate & Download",
    data=lambda: build_txt_bytes(out_lang),  # generate only on click [web:78]
    file_name=fname,
    mime="text/plain",
    use_container_width=True,
    key="dl",
)
