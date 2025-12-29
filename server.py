# server.py (ULTRA-SIMPLE + local test mode)
# - API:  POST /infer  (upload wav)
# - Local test: python server.py --wav path/to/test.wav
#
# Install:
#   pip install fastapi uvicorn librosa numpy torch
#   (plus PantoMatrix deps per their setup)
#
# Run server:
#   uvicorn server:app --host 127.0.0.1 --port 8000
#
# Local test (no server needed):
#   python server.py --wav ./examples/audio/sample.wav

import io
import argparse
import numpy as np
import torch
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Adjust import to match your repo layout
from models.emage_audio import EmageAudioModel

app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "H-Liu1997/emage_audio"
model = None


def load_model():
    global model
    if model is None:
        model = EmageAudioModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
        torch.set_grad_enabled(False)
        print("Loaded", MODEL_ID, "on", DEVICE)
    return model


def infer_from_audio_np(audio_np: np.ndarray, sr: int):
    """
    audio_np: mono waveform float32
    sr: sample rate of audio_np
    returns: (fps_hint, motion_flat[F, D])
    """
    m = load_model()
    target_sr = getattr(getattr(m, "cfg", None), "audio_sr", 16000)

    if sr != target_sr:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    audio = torch.from_numpy(audio_np).float().to(DEVICE).unsqueeze(0)  # (1, T)

    with torch.no_grad():
        out = m(audio)

    if "motion_axis_angle" not in out:
        raise RuntimeError(f"Unexpected keys: {list(out.keys())}")

    motion = out["motion_axis_angle"].detach().cpu().numpy()

    # Normalize to (F, D)
    if motion.ndim == 4:      # (1, F, J, 3)
        motion = motion[0].reshape(motion.shape[1], -1)
    elif motion.ndim == 3:    # (1, F, D)
        motion = motion[0]
    else:
        raise RuntimeError(f"Unexpected motion shape: {motion.shape}")

    fps_hint = 30  # treat as hint
    return fps_hint, motion


@app.on_event("startup")
def _startup():
    load_model()


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Upload a .wav")

    audio_bytes = await file.read()

    m = load_model()
    target_sr = getattr(getattr(m, "cfg", None), "audio_sr", 16000)

    # Load wav bytes -> mono waveform
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=target_sr, mono=True)
    fps, motion = infer_from_audio_np(y.astype(np.float32), sr)

    return JSONResponse({
        "fps": fps,
        "motion_axis_angle_flat": motion.tolist(),  # [F][D]
        "frames": int(motion.shape[0]),
        "dims": int(motion.shape[1]),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, default=None, help="Run a quick local inference on this wav file")
    parser.add_argument("--save_npz", type=str, default=None, help="Optional: save output as .npz")
    args = parser.parse_args()

    if not args.wav:
        print("No --wav provided. Start the API with: uvicorn server:app --host 127.0.0.1 --port 8000")
        return

    # Local test
    y, sr = librosa.load(args.wav, sr=None, mono=True)
    fps, motion = infer_from_audio_np(y.astype(np.float32), sr)
    print(f"OK. fps={fps}, frames={motion.shape[0]}, dims={motion.shape[1]}")

    if args.save_npz:
        np.savez(args.save_npz, motion_axis_angle_flat=motion, fps=fps)
        print("Saved:", args.save_npz)


if __name__ == "__main__":
    main()
