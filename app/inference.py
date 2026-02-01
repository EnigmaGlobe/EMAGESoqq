# app/inference.py
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import logging
from typing import Tuple

from app.config import DEFAULT_FPS_HINT
from app.deps import load_bundle

def infer_from_audio_np(audio_np: np.ndarray, sr: int) -> Tuple[int, np.ndarray]:
    """
    audio_np: mono waveform float32
    sr: sample rate
    returns: (fps_hint, motion_flat[F, D])
    """
    if not isinstance(audio_np, np.ndarray):
        raise ValueError("audio_np must be a numpy.ndarray")
    if audio_np.ndim != 1:
        raise ValueError(f"audio_np must be 1D (mono). Got shape={audio_np.shape}")
    if not np.isfinite(audio_np).all():
        raise ValueError("audio_np contains NaN or Inf values")

    bundle = load_bundle()
    m = bundle.model
    motion_vq = bundle.motion_vq
    device = bundle.device

    target_sr = getattr(getattr(m, "cfg", None), "audio_sr", 16000)
    if sr != target_sr:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    audio = torch.from_numpy(audio_np).float().to(device).unsqueeze(0)  # (1, T)

    with torch.no_grad():
        speaker_id = torch.zeros(1, 1).long().to(device)
        latent_out = m.inference(audio, speaker_id, motion_vq, masked_motion=None, mask=None)

    cfg = getattr(m, "cfg", None)

    face_latent = latent_out.get("rec_face") if (cfg is not None and getattr(cfg, "lf", 0) > 0 and getattr(cfg, "cf", 0) == 0) else None
    upper_latent = latent_out.get("rec_upper") if (cfg is not None and getattr(cfg, "lu", 0) > 0 and getattr(cfg, "cu", 0) == 0) else None
    hands_latent = latent_out.get("rec_hands") if (cfg is not None and getattr(cfg, "lh", 0) > 0 and getattr(cfg, "ch", 0) == 0) else None
    lower_latent = latent_out.get("rec_lower") if (cfg is not None and getattr(cfg, "ll", 0) > 0 and getattr(cfg, "cl", 0) == 0) else None

    face_index = upper_index = hands_index = lower_index = None
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

    if motion.ndim == 4:      # (1, F, J, 3)
        motion = motion[0].reshape(motion.shape[1], -1)
    elif motion.ndim == 3:    # (1, F, D)
        motion = motion[0]
    else:
        raise RuntimeError(f"Unexpected motion shape: {motion.shape}")

    return DEFAULT_FPS_HINT, motion
