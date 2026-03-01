import React, { useEffect, useMemo, useRef, useState } from "react";
import { Streamlit, ComponentProps } from "streamlit-component-lib";

function floatToPCM16(samples: Float32Array): Int16Array {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    out[i] = s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff);
  }
  return out;
}

function encodeWavPCM16Mono(samples: Float32Array, sampleRate: number): Uint8Array {
  const pcm16 = floatToPCM16(samples);
  const numFrames = pcm16.length;
  const buffer = new ArrayBuffer(44 + numFrames * 2);
  const view = new DataView(buffer);

  const writeString = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };

  let o = 0;
  writeString(o, "RIFF"); o += 4;
  view.setUint32(o, 36 + numFrames * 2, true); o += 4;
  writeString(o, "WAVE"); o += 4;

  writeString(o, "fmt "); o += 4;
  view.setUint32(o, 16, true); o += 4;        // PCM fmt chunk size
  view.setUint16(o, 1, true); o += 2;         // PCM
  view.setUint16(o, 1, true); o += 2;         // mono
  view.setUint32(o, sampleRate, true); o += 4;
  view.setUint32(o, sampleRate * 2, true); o += 4; // byte rate
  view.setUint16(o, 2, true); o += 2;         // block align
  view.setUint16(o, 16, true); o += 2;        // bits/sample

  writeString(o, "data"); o += 4;
  view.setUint32(o, numFrames * 2, true); o += 4;

  let dataOffset = 44;
  for (let i = 0; i < numFrames; i++) {
    view.setInt16(dataOffset + i * 2, pcm16[i], true);
  }

  return new Uint8Array(buffer);
}

type RecorderState = {
  ctx: AudioContext;
  source: MediaStreamAudioSourceNode;
  processor: ScriptProcessorNode;
  stream: MediaStream;
  chunks: Float32Array[];
};

export default function MicButton(_props: ComponentProps) {
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const recRef = useRef<RecorderState | null>(null);

  useEffect(() => {
    Streamlit.setFrameHeight(170);
  }, []);

  const start = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    const ctx = new AudioContext();
    const source = ctx.createMediaStreamSource(stream);
    const processor = ctx.createScriptProcessor(4096, 1, 1);

    const chunks: Float32Array[] = [];
    processor.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      chunks.push(new Float32Array(input));
    };

    source.connect(processor);
    processor.connect(ctx.destination);

    recRef.current = { ctx, source, processor, stream, chunks };
  };

  const stop = async () => {
    const r = recRef.current;
    if (!r) return null;

    r.processor.disconnect();
    r.source.disconnect();
    r.stream.getTracks().forEach((t) => t.stop());

    const sampleRate = r.ctx.sampleRate;
    await r.ctx.close();

    const total = r.chunks.reduce((a, c) => a + c.length, 0);
    const samples = new Float32Array(total);
    let off = 0;
    for (const c of r.chunks) { samples.set(c, off); off += c.length; }

    recRef.current = null;

    const wav = encodeWavPCM16Mono(samples, sampleRate);
    return { wavBytes: wav, sampleRate };
  };

  const onToggle = async () => {
    if (busy) return;
    setBusy(true);
    try {
      if (!recording) {
        await start();
        setRecording(true);
        Streamlit.setComponentValue({ status: "recording" });
      } else {
        const res = await stop();
        setRecording(false);

        if (res && res.wavBytes.length > 44) {
          // Send WAV as an array of ints (Streamlit component values must be JSON-serializable)
          Streamlit.setComponentValue({
            status: "stopped",
            wav_bytes: Array.from(res.wavBytes),
            mime: "audio/wav",
            sample_rate: res.sampleRate,
            sample_width: 2,
            channels: 1
          });
        } else {
          Streamlit.setComponentValue({ status: "stopped", wav_bytes: null });
        }
      }
    } catch (e: any) {
      setRecording(false);
      Streamlit.setComponentValue({ status: "error", error: String(e) });
    } finally {
      setBusy(false);
    }
  };

  const size = 86;

  const ringStyle: React.CSSProperties = useMemo(() => ({
    position: "absolute",
    inset: -10,
    borderRadius: "50%",
    border: "2px solid rgba(139, 92, 246, 0.95)",
    opacity: recording ? 1 : 0,
    animation: recording ? "pulse 1.1s ease-out infinite" : "none",
    pointerEvents: "none"
  }), [recording]);

  const btnStyle: React.CSSProperties = useMemo(() => ({
    width: size,
    height: size,
    borderRadius: "50%",
    border: "2px solid rgba(255,255,255,0.18)",
    background: recording ? "rgba(139,92,246,0.30)" : "rgba(255,255,255,0.06)",
    color: "white",
    fontSize: 30,
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: busy ? "not-allowed" : "pointer",
    position: "relative",
    userSelect: "none",
    transform: recording ? "scale(1.02)" : "scale(1.0)",
    transition: "transform 120ms ease"
  }), [recording, busy]);

  const iconStyle: React.CSSProperties = useMemo(() => ({
    animation: recording ? "bob 0.6s ease-in-out infinite alternate" : "none"
  }), [recording]);

  return (
    <div style={{display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", height:150}}>
      <style>{`
        @keyframes pulse {
          0% { transform: scale(1); opacity: 0.85; }
          100% { transform: scale(1.28); opacity: 0; }
        }
        @keyframes bob {
          0% { transform: translateY(0px); }
          100% { transform: translateY(-2px); }
        }
      `}</style>

      <div style={{position:"relative"}}>
        <div style={ringStyle}></div>
        <div style={btnStyle} onClick={onToggle} role="button" aria-label="microphone-toggle">
          <span style={iconStyle}>{recording ? "■" : "🎤"}</span>
        </div>
      </div>

      <div style={{marginTop: 10, fontSize: 12, opacity: 0.8}}>
        {recording ? "Recording…" : "Tap to record"}
      </div>
    </div>
  );
}
