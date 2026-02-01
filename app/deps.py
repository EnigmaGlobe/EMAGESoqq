# app/deps.py
import logging
import torch
from dataclasses import dataclass
from typing import Optional, Tuple

from app.config import MODEL_ID
from models.emage_audio import EmageAudioModel

@dataclass
class ModelBundle:
    model: EmageAudioModel
    motion_vq: torch.nn.Module
    device: torch.device

_bundle: Optional[ModelBundle] = None

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_bundle() -> ModelBundle:
    global _bundle
    if _bundle is not None:
        return _bundle

    device = get_device()
    logging.info("Loading model and VQ components...")

    model = EmageAudioModel.from_pretrained(MODEL_ID).to(device).eval()

    from models.emage_audio import EmageVQVAEConv, EmageVAEConv, EmageVQModel

    face_motion_vq = EmageVQVAEConv.from_pretrained(MODEL_ID, subfolder="emage_vq/face").to(device)
    upper_motion_vq = EmageVQVAEConv.from_pretrained(MODEL_ID, subfolder="emage_vq/upper").to(device)
    lower_motion_vq = EmageVQVAEConv.from_pretrained(MODEL_ID, subfolder="emage_vq/lower").to(device)
    hands_motion_vq = EmageVQVAEConv.from_pretrained(MODEL_ID, subfolder="emage_vq/hands").to(device)
    global_motion_ae = EmageVAEConv.from_pretrained(MODEL_ID, subfolder="emage_vq/global").to(device)

    motion_vq = EmageVQModel(
        face_model=face_motion_vq,
        upper_model=upper_motion_vq,
        lower_model=lower_motion_vq,
        hands_model=hands_motion_vq,
        global_model=global_motion_ae,
    ).to(device).eval()

    torch.set_grad_enabled(False)

    logging.info("Loaded %s on %s", MODEL_ID, device)
    _bundle = ModelBundle(model=model, motion_vq=motion_vq, device=device)
    return _bundle
