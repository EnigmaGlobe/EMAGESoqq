# app/routes.py
import io
from io import BytesIO
import time
import logging
import struct

import numpy as np
import librosa

from fastapi import APIRouter, File, UploadFile, HTTPException, Body, Query
from fastapi.responses import JSONResponse, Response

from app.config import MAX_UPLOAD_BYTES, MAX_WAV_SECONDS
from app.deps import load_bundle
from app.inference import infer_from_audio_np

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe: process is up and serving HTTP."""
    return {"status": "ok"}


@router.get("/ready")
async def ready() -> JSONResponse:
    """Readiness probe: model dependencies can be loaded and served."""
    try:
        load_bundle()
        return JSONResponse({"status": "ready"}, status_code=200)
    except Exception:
        logging.exception("readiness check failed")
        return JSONResponse({"status": "not_ready"}, status_code=503)

# -----------------------------------------------------------------------------
# Binary wire format (little-endian)
# -----------------------------------------------------------------------------
# Header (12 bytes):
#   uint32 fps
#   uint32 frames
#   uint32 dims
# Payload:
#   float32[frames * dims] in row-major order (frame0 all dims, frame1 all dims, ...)
#
# Unity can decode with BitConverter + Buffer.BlockCopy into float[].
# -----------------------------------------------------------------------------
@router.post("/infer_pcm16_raw")
async def infer_pcm16_raw(
    pcm16_bytes: bytes = Body(..., media_type="application/octet-stream"),
    sr: int = Query(24000),
    channels: int = Query(1),
):
    # Size guard (bytes)
    if len(pcm16_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file is too large")

    # Decode PCM16 LE -> float32 [-1, 1]
    x = np.frombuffer(pcm16_bytes, dtype="<i2")  # little-endian int16
    if channels == 2:
        if x.size % 2 != 0:
            raise HTTPException(status_code=400, detail="Invalid stereo PCM length")
        x = x.reshape(-1, 2).mean(axis=1)
    elif channels != 1:
        raise HTTPException(status_code=400, detail="channels must be 1 or 2")

    y = (x.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

    duration_sec = float(len(y)) / float(sr) if len(y) else 0.0
    if duration_sec > MAX_WAV_SECONDS:
        raise HTTPException(status_code=413, detail="Audio too long")

    fps, motion = infer_from_audio_np(y, sr)
    blob = _pack_motion_npz(fps, np.asarray(motion, dtype=np.float32, order="C"))
    return Response(content=blob, media_type="application/octet-stream")

def _decode_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int, float]:
    """WAV bytes -> (mono float32 waveform y, sample rate sr, duration_sec)."""
    nbytes = len(wav_bytes)
    if nbytes > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file is too large")

    try:
        y, sr = librosa.load(io.BytesIO(wav_bytes), sr=None, mono=True)
    except Exception:
        logging.warning("infer: librosa decode failed (bytes=%d)", nbytes, exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid or corrupted WAV file")

    if y is None or sr is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupted WAV file")

    y = np.asarray(y, dtype=np.float32)
    duration_sec = float(len(y)) / float(sr) if len(y) else 0.0

    if duration_sec > MAX_WAV_SECONDS:
        raise HTTPException(status_code=413, detail="Audio too long")

    return y, int(sr), duration_sec


def _infer_motion_from_wav_bytes(wav_bytes: bytes) -> tuple[int, np.ndarray]:
    """Shared core: WAV bytes -> librosa decode -> infer -> (fps, motion float32 ndarray)."""
    nbytes = len(wav_bytes)
    logging.info("infer: received wav bytes=%d", nbytes)

    t0 = time.perf_counter()
    y, sr, duration_sec = _decode_wav_bytes(wav_bytes)

    logging.info("infer: decoded sr=%d samples=%d duration=%.3fs", sr, int(len(y)), float(duration_sec))

    try:
        fps, motion = infer_from_audio_np(y.astype(np.float32), int(sr))
    except RuntimeError:
        logging.exception("infer: model inference failed")
        raise HTTPException(status_code=500, detail="Model inference failed")

    # Ensure contiguous float32 (important for fast tobytes)
    motion = np.asarray(motion, dtype=np.float32, order="C")

    frames = int(motion.shape[0])
    dims = int(motion.shape[1]) if motion.ndim == 2 else int(motion.size)
    dt = time.perf_counter() - t0
    logging.info("infer: done fps=%d frames=%d dims=%d elapsed=%.3fs", int(fps), frames, dims, dt)

    return int(fps), motion


def _pack_motion_f32(fps: int, motion_f32: np.ndarray) -> bytes:
    """Pack to: [u32 fps][u32 frames][u32 dims][float32 payload]."""
    if motion_f32.ndim != 2:
        raise HTTPException(status_code=500, detail=f"Unexpected motion shape: {motion_f32.shape}")

    frames = int(motion_f32.shape[0])
    dims = int(motion_f32.shape[1])

    header = struct.pack("<III", int(fps), frames, dims)
    payload = motion_f32.astype(np.float32, copy=False).tobytes(order="C")
    return header + payload


def _pack_motion_npz(fps: int, motion_f32: np.ndarray) -> bytes:
    """Pack to an in-memory .npz containing the motion output."""
    if motion_f32.ndim != 2:
        raise HTTPException(status_code=500, detail=f"Unexpected motion shape: {motion_f32.shape}")

    buffer = BytesIO()
    np.savez(
        buffer,
        fps=np.int32(fps),
        frames=np.int32(motion_f32.shape[0]),
        dims=np.int32(motion_f32.shape[1]),
        motion_axis_angle_flat=motion_f32.astype(np.float32, copy=False),
    )
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@router.post("/infer")
async def infer(file: UploadFile = File(...)):
    """
    Multipart upload route: expects a .wav file, returns JSON (legacy).
    """
    filename = (file.filename or "").lower()
    if not filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav uploads are allowed")

    wav_bytes = await file.read()

    try:
        fps, motion = _infer_motion_from_wav_bytes(wav_bytes)
        frames = int(motion.shape[0])
        dims = int(motion.shape[1])

        # JSON is huge but kept for compatibility
        return JSONResponse(
            {
                "fps": int(fps),
                "motion_axis_angle_flat": motion.tolist(),
                "frames": frames,
                "dims": dims,
            }
        )
    except HTTPException:
        raise
    except (ValueError, TypeError, np.linalg.LinAlgError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logging.exception("Unexpected error")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/infer_bytes")
async def infer_bytes(
    wav_bytes: bytes = Body(..., media_type="application/octet-stream"),
):
    """
    Raw-bytes route: expects WAV bytes in request body, returns JSON (legacy).
    """
    try:
        fps, motion = _infer_motion_from_wav_bytes(wav_bytes)
        frames = int(motion.shape[0])
        dims = int(motion.shape[1])

        return JSONResponse(
            {
                "fps": int(fps),
                "motion_axis_angle_flat": motion.tolist(),
                "frames": frames,
                "dims": dims,
            }
        )
    except HTTPException:
        raise
    except (ValueError, TypeError, np.linalg.LinAlgError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logging.exception("Unexpected error")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/infer_raw")
async def infer_raw(
    wav_bytes: bytes = Body(..., media_type="application/octet-stream"),
):
    """
    Raw-bytes in, RAW float32 out.

    Response content-type: application/octet-stream
    Format: [u32 fps][u32 frames][u32 dims][float32 payload]
    """
    try:
        fps, motion = _infer_motion_from_wav_bytes(wav_bytes)
        blob = _pack_motion_f32(fps, motion)

        # Response (NOT JSON)
        return Response(content=blob, media_type="application/octet-stream")
    except HTTPException:
        raise
    except (ValueError, TypeError, np.linalg.LinAlgError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logging.exception("Unexpected error")
        raise HTTPException(status_code=500, detail="Internal Server Error")
