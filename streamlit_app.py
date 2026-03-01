# streamlit_app.py
# speak2list_kannada — Streamlit + streamlit-mic-recorder + Whisper + soundfile (no ffmpeg)

import io
import re
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import streamlit as st
from om_transliterator import Transliterator  # Kannada -> Latin transliteration [web:218]
from streamlit_mic_recorder import mic_recorder  # click to start, click to stop [web:89]
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="speak2list_kannada", layout="centered")

st.markdown(
    """
<style>
/* Center everything a bit */
.block-container { padding-top: 1.5rem; }

/* Make the mic recorder button look circular.
   streamlit-mic-recorder renders a Streamlit component; exact DOM can vary by version.
   This targets common button elements inside the component container. */
div[data-testid="stMicRecorder"] button,
div[data-testid="stMicRecorder"] [role="button"],
.stMicRecorder button,
.stMicRecorder [role="button"] {
  width: 84px !important;
  height: 84px !important;
  border-radius: 50% !important;
  padding: 0 !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  font-size: 28px !important;
  border: 2px solid rgba(255,255,255,0.18) !important;
}

/* If the component uses Streamlit button styles inside */
div[data-testid="stMicRecorder"] button:hover,
div[data-testid="stMicRecorder"] [role="button"]:hover {
  filter: brightness(1.05);
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("speak2list_kannada")
st.caption("Android Chrome • Record Kannada voice • Download shopping list")

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
# Session state
# ----------------------------
if "wav_clips" not in st.session_state:
    st.session_state.wav_clips: List[Tuple[np.ndarray, int]] = []  # (audio_float32, sr)

if "transcript_kn" not in st.session_state:
    st.session_state.transcript_kn = ""

if "shopping_items" not in st.session_state:
    st.session_state.shopping_items: List[Tuple[str, Optional[str]]] = []

if "list_text" not in st.session_state:
    st.session_state.list_text = ""

if "last_lang" not in st.session_state:
    st.session_state.last_lang = "Kannada"

# ----------------------------
# Helpers
# ----------------------------
def reset_all():
    st.session_state.wav_clips = []
    st.session_state.transcript_kn = ""
    st.session_state.shopping_items = []
    st.session_state.list_text = ""
    st.session_state.last_lang = "Kannada"


def decode_wav_bytes_to_float32(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    # streamlit-mic-recorder returns mono wav bytes by default [web:89]
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")  # soundfile supports file-like objects [web:248]
    if data.ndim > 1:
        data = data.mean(axis=1)  # to mono
    return data.astype(np.float32), int(sr)


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
    # Lightweight “relevant part” extractor without ffmpeg/pydub:
    # keep frames whose RMS energy is above a threshold and pad neighbors.
    if len(x) == 0:
        return x

    frame_ms = 30
    hop_ms = 10
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if frame <= 0 or hop <= 0 or len(x) < frame:
        return x

    eps = 1e-8
    rms = []
    for i in range(0, len(x) - frame + 1, hop):
        w = x[i : i + frame]
        rms.append(float(np.sqrt(np.mean(w * w) + eps)))
    rms = np.array(rms, dtype=np.float32)

    if rms.size == 0:
        return x

    # Adaptive threshold: slightly above the 20th percentile energy
    thr = float(np.percentile(rms, 20)) * 2.5
    speech_mask = rms > max(thr, 0.01)

    if not speech_mask.any():
        return x  # fallback: keep all

    # Expand mask by +/- 5 frames (~50ms each side with hop=10ms)
    k = 5
    expanded = np.copy(speech_mask)
    for i in range(speech_mask.size):
        if speech_mask[i]:
            lo = max(0, i - k)
            hi = min(speech_mask.size, i + k + 1)
            expanded[lo:hi] = True

    # Convert mask back to sample indices
    keep = []
    for idx, keep_flag in enumerate(expanded):
        if keep_flag:
            start = idx * hop
            end = start + frame
            keep.append((start, min(end, len(x))))

    # Merge overlapping intervals
    keep.sort()
    merged = []
    cur_s, cur_e = keep[0]
    for s, e in keep[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    out = np.concatenate([x[s:e] for s, e in merged]).astype(np.float32)
    return out


def transcribe_kn_from_wav16(wav16: np.ndarray) -> str:
    if wav16 is None or len(wav16) < 1600:  # ~0.1s at 16k
        return ""
    inputs = processor(wav16, sampling_rate=16000, return_tensors="pt")
    predicted_ids = model.generate(inputs.input_features)
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text.strip()


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
    if not st.session_state.wav_clips:
        st.session_state.transcript_kn = ""
        st.session_state.shopping_items = []
        st.session_state.list_text = ""
        st.session_state.last_lang = out_lang
        return b""

    # Concatenate all recorded clips
    wav = np.concatenate([x for (x, _sr) in st.session_state.wav_clips]).astype(np.float32)
    # Resample each clip already stored with its SR? (We stored raw per-clip; simplest: resample per-clip first)
    # Here we assume each clip could have different sr; resample individually then concat:
    wav16 = np.concatenate([resample_to_16k(x, sr) for (x, sr) in st.session_state.wav_clips]).astype(np.float32)

    # Keep only "relevant" speech-like parts (simple VAD)
    speech16 = simple_vad_keep_speech(wav16, sr=16000)

    transcript_kn = transcribe_kn_from_wav16(speech16)
    shopping_items = extract_items(transcript_kn)
    list_text = format_list(shopping_items, out_lang)

    st.session_state.transcript_kn = transcript_kn
    st.session_state.shopping_items = shopping_items
    st.session_state.list_text = list_text
    st.session_state.last_lang = out_lang

    return (list_text + "\n").encode("utf-8")


# ----------------------------
# Controls (main screen, no sidebar)
# ----------------------------
top = st.columns([1.2, 1, 1])
with top[0]:
    out_lang = st.selectbox(
        "Download language",
        ["Kannada", "English (transliteration)"],
        index=0 if st.session_state.last_lang == "Kannada" else 1,
    )
with top[1]:
    st.write("")
    if st.button("Clear / start fresh", use_container_width=True):
        reset_all()
with top[2]:
    st.write("")
    date_tag = datetime.now().strftime("%Y-%m-%d")
    fname = f"speak2list_kannada_{date_tag}.txt"
    st.download_button(
        label="Generate & Download",
        data=lambda: build_txt_bytes(out_lang),  # only generate selected language [web:78]
        file_name=fname,
        mime="text/plain",
        use_container_width=True,
        key="dl",
    )

st.divider()

# ----------------------------
# Recorder UI
# ----------------------------
st.subheader("Record")
st.caption("Tap to start, tap again to stop. You can record multiple clips; they will be combined.")

# Mic symbol: Streamlit buttons support icon, but this component provides its own UI.
# We set prompts to a mic and stop icon for clarity. [web:89]
audio = mic_recorder(
    start_prompt="🎤",
    stop_prompt="⏹️",
    just_once=True,
    use_container_width=False,
    key="recorder",
)

if audio is not None:
    wav, sr = decode_wav_bytes_to_float32(audio["bytes"])
    st.session_state.wav_clips.append((wav, sr))
    total_s = sum(len(x) / s for (x, s) in st.session_state.wav_clips if len(x) > 0 and s > 0)
    st.success(f"Clip added. Total recorded: ~{total_s:.1f}s")

st.divider()

# ----------------------------
# Preview
# ----------------------------
st.subheader("Preview")
st.text_area("Transcript (Kannada)", value=st.session_state.transcript_kn, height=120)
st.code(st.session_state.list_text or "(No items yet)", language="text")
