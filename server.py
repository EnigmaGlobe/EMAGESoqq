import io
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import logging
from models.emage_audio import EmageAudioModel

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# FastAPI app
app = FastAPI()

# Device setup (use CUDA if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "H-Liu1997/emage_audio"
model = None
motion_vq = None

# Upload limits
# Maximum binary upload size accepted (bytes)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
# Maximum audio duration (seconds) to process
MAX_WAV_SECONDS = 60

# Function to load the model
def load_model():
    global model
    global motion_vq
    if model is None or motion_vq is None:
        logging.info("Loading model and VQ components...")
        # Load audio model
        model = EmageAudioModel.from_pretrained(MODEL_ID).to(DEVICE).eval()

        # Load motion VQ components required by model.inference()
        from models.emage_audio import EmageVQVAEConv, EmageVAEConv, EmageVQModel

        face_motion_vq = EmageVQVAEConv.from_pretrained(MODEL_ID, subfolder="emage_vq/face").to(DEVICE)
        upper_motion_vq = EmageVQVAEConv.from_pretrained(MODEL_ID, subfolder="emage_vq/upper").to(DEVICE)
        lower_motion_vq = EmageVQVAEConv.from_pretrained(MODEL_ID, subfolder="emage_vq/lower").to(DEVICE)
        hands_motion_vq = EmageVQVAEConv.from_pretrained(MODEL_ID, subfolder="emage_vq/hands").to(DEVICE)
        global_motion_ae = EmageVAEConv.from_pretrained(MODEL_ID, subfolder="emage_vq/global").to(DEVICE)

        motion_vq = EmageVQModel(
            face_model=face_motion_vq,
            upper_model=upper_motion_vq,
            lower_model=lower_motion_vq,
            hands_model=hands_motion_vq,
            global_model=global_motion_ae,
        ).to(DEVICE)
        motion_vq.eval()

        torch.set_grad_enabled(False)
        logging.info(f"Loaded {MODEL_ID} and motion VQ on {DEVICE}")
    return model

# Function to process and run inference on audio
def infer_from_audio_np(audio_np: np.ndarray, sr: int):
    """
    audio_np: mono waveform float32
    sr: sample rate of audio_np
    returns: (fps_hint, motion_flat[F, D])
    """
    # Basic validation of the numpy input to avoid surprising numpy internals
    if not isinstance(audio_np, np.ndarray):
        raise ValueError("audio_np must be a numpy.ndarray")
    if audio_np.ndim != 1:
        raise ValueError(f"audio_np must be 1D (mono). Got shape={audio_np.shape}")
    if not np.isfinite(audio_np).all():
        raise ValueError("audio_np contains NaN or Inf values")

    m = load_model()  # Ensure the model is loaded
    target_sr = getattr(getattr(m, "cfg", None), "audio_sr", 16000)

    # Resample the audio if needed
    if sr != target_sr:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Convert to PyTorch tensor
    audio = torch.from_numpy(audio_np).float().to(DEVICE).unsqueeze(0)  # (1, T)

    try:
        with torch.no_grad():
            # Prepare a default speaker id tensor
            speaker_id = torch.zeros(1, 1).long().to(DEVICE)

            # Perform inference to get latent/class outputs
            latent_out = m.inference(audio, speaker_id, motion_vq, masked_motion=None, mask=None)
    except RuntimeError:
        # Torch runtime errors during inference are likely model/input problems
        logging.exception("Runtime error during model inference")
        raise

    # Convert latent/class outputs into decoded motion via the motion VQ model
    cfg = getattr(m, "cfg", None)

    face_latent = latent_out.get("rec_face") if (cfg is not None and getattr(cfg, "lf", 0) > 0 and getattr(cfg, "cf", 0) == 0) else None
    upper_latent = latent_out.get("rec_upper") if (cfg is not None and getattr(cfg, "lu", 0) > 0 and getattr(cfg, "cu", 0) == 0) else None
    hands_latent = latent_out.get("rec_hands") if (cfg is not None and getattr(cfg, "lh", 0) > 0 and getattr(cfg, "ch", 0) == 0) else None
    lower_latent = latent_out.get("rec_lower") if (cfg is not None and getattr(cfg, "ll", 0) > 0 and getattr(cfg, "cl", 0) == 0) else None

    face_index = None
    upper_index = None
    hands_index = None
    lower_index = None
    try:
        if cfg is not None and getattr(cfg, "cf", 0) > 0:
            face_index = torch.max(F.log_softmax(latent_out["cls_face"], dim=2), dim=2)[1]
        if cfg is not None and getattr(cfg, "cu", 0) > 0:
            upper_index = torch.max(F.log_softmax(latent_out["cls_upper"], dim=2), dim=2)[1]
        if cfg is not None and getattr(cfg, "ch", 0) > 0:
            hands_index = torch.max(F.log_softmax(latent_out["cls_hands"], dim=2), dim=2)[1]
        if cfg is not None and getattr(cfg, "cl", 0) > 0:
            lower_index = torch.max(F.log_softmax(latent_out["cls_lower"], dim=2), dim=2)[1]
    except KeyError:
        # If classification outputs are missing, fall back to latents-only decode
        logging.debug("Class outputs missing; proceeding without indices for decode")

    decode_dict = motion_vq.decode(
        face_latent=face_latent,
        upper_latent=upper_latent,
        lower_latent=lower_latent,
        hands_latent=hands_latent,
        face_index=face_index,
        upper_index=upper_index,
        lower_index=lower_index,
        hands_index=hands_index,
        get_global_motion=False,
    )

    if "motion_axis_angle" not in decode_dict:
        raise RuntimeError(f"Unexpected keys in VQ decode output: {list(decode_dict.keys())}")

    motion = decode_dict["motion_axis_angle"].detach().cpu().numpy()

    # Normalize motion to (F, D)
    if motion.ndim == 4:      # (1, F, J, 3)
        motion = motion[0].reshape(motion.shape[1], -1)
    elif motion.ndim == 3:    # (1, F, D)
        motion = motion[0]
    else:
        raise RuntimeError(f"Unexpected motion shape: {motion.shape}")

    fps_hint = 30  # fps is treated as a hint
    return fps_hint, motion

# API Endpoint for inference
@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    """
    Accepts either a `.wav` upload (multipart/form) or a `.npz` file containing
    precomputed motion arrays. Returns JSON with `motion_axis_angle_flat`,
    `frames`, `dims`, and `fps`.
    """

    filename = (file.filename or "").lower()

    # Basic upload size guard
    audio_bytes = await file.read()
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        logging.warning("Upload too large: %d bytes", len(audio_bytes))
        raise HTTPException(status_code=413, detail="Uploaded file is too large")

    # Handle NPZ uploads: convert to JSON response if it contains motion
    if filename.endswith(".npz"):
        try:
            buffer = io.BytesIO(audio_bytes)
            with np.load(buffer, allow_pickle=False) as data:
                # Look for common keys produced by this project
                if "motion_axis_angle_flat" in data:
                    motion = np.asarray(data["motion_axis_angle_flat"])
                elif "motion" in data:
                    motion = np.asarray(data["motion"])
                elif "motion_axis_angle" in data:
                    motion = np.asarray(data["motion_axis_angle"])
                else:
                    raise ValueError(".npz does not contain motion data")

            if motion.ndim != 2:
                # Try to coerce common shapes
                if motion.ndim == 1:
                    motion = motion.reshape(-1, 1)
                else:
                    raise ValueError(f"Unexpected motion array shape: {motion.shape}")

            return JSONResponse({
                "fps": int(data.get("fps", 30)),
                "motion_axis_angle_flat": motion.tolist(),
                "frames": int(motion.shape[0]),
                "dims": int(motion.shape[1]),
            })
        except (ValueError, OSError, KeyError) as e:
            logging.warning("Bad .npz upload: %s", e)
            raise HTTPException(status_code=400, detail=str(e))
        except Exception:
            logging.exception("Failed to process .npz upload")
            raise HTTPException(status_code=500, detail="Failed to process .npz file")

    # Only allow wav uploads for inference
    if not filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav or .npz uploads are allowed")

    try:
        # Load the model lazily (startup also loads it)
        _ = load_model()

        # Use librosa to read audio. Wrap and translate errors to 400.
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        except Exception as e:
            logging.warning("Failed to decode WAV upload: %s", e)
            raise HTTPException(status_code=400, detail="Invalid or corrupted WAV file")

        # Duration guard
        duration_sec = float(len(y)) / float(sr) if sr and len(y) else 0.0
        if duration_sec > MAX_WAV_SECONDS:
            logging.warning("WAV duration too long: %s seconds", duration_sec)
            raise HTTPException(status_code=413, detail="Audio too long")

        # Run inference (this will raise RuntimeError on model failures)
        fps, motion = infer_from_audio_np(y.astype(np.float32), sr)

        return JSONResponse({
            "fps": fps,
            "motion_axis_angle_flat": motion.tolist(),
            "frames": int(motion.shape[0]),
            "dims": int(motion.shape[1]),
        })

    except (ValueError, TypeError, np.linalg.LinAlgError) as e:
        logging.warning("Bad request during file processing: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Re-raise HTTPExceptions created above
        raise
    except RuntimeError as e:
        logging.exception("Model inference failed: %s", e)
        raise HTTPException(status_code=500, detail="Model inference failed")
    except Exception as e:
        logging.exception("Unexpected error processing file: %s", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Startup event: Load the model when the server starts
@app.on_event("startup")
def _startup():
    load_model()

# Local testing function (for testing with a .wav file directly without the API)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True, help="Path to the .wav file for local inference")
    parser.add_argument("--save_npz", type=str, help="Optional: Save the output as a .npz file")
    args = parser.parse_args()

    # Local test
    try:
        y, sr = librosa.load(args.wav, sr=None, mono=True)
        fps, motion = infer_from_audio_np(y.astype(np.float32), sr)
        print(f"OK. fps={fps}, frames={motion.shape[0]}, dims={motion.shape[1]}")

        if args.save_npz:
            np.savez(args.save_npz, motion_axis_angle_flat=motion, fps=fps)
            print(f"Saved output to: {args.save_npz}")

    except (ValueError, TypeError, np.linalg.LinAlgError) as e:
        print(f"Input error in local test: {e}")
    except RuntimeError as e:
        print(f"Model/runtime error in local test: {e}")
    except Exception as e:
        print(f"Unexpected error in local test: {e}")

if __name__ == "__main__":
    # Check if running locally or as a FastAPI server
    import sys
    if '--wav' in sys.argv:
        main()  # Run local test
    else:
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000)  # Start FastAPI server
