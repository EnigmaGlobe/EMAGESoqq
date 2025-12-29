# test_emage_audio_windows.py
# Windows-friendly EMAGE demo:
# - Always generates .npz motion outputs.
# - Visualization behavior:
#     * --visualization --nopytorch3d : SKIPS 2D (because repo's render2d imports pytorch3d on Windows)
#     * --visualization (no nopytorch3d): tries 2D via render2d (requires pytorch3d) and 3D via fast_render (OpenGL)
# - By default we DISABLE fast_render to avoid EGL/OpenGL issues on Windows.

import os
import argparse
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.io import write_video
import librosa

from emage_utils.motion_io import beat_format_save

# --- Rendering is optional; disable by default on Windows ---
# If you later get OpenGL working, you can switch to try/except import below.
fast_render = None  # keep disabled on Windows

from models.emage_audio import EmageAudioModel, EmageVQVAEConv, EmageVAEConv, EmageVQModel


def inference(model, motion_vq, audio_path, device, save_folder, sr, pose_fps):
    audio, _ = librosa.load(audio_path, sr=sr)
    audio = torch.from_numpy(audio).to(device).unsqueeze(0)
    speaker_id = torch.zeros(1, 1).long().to(device)

    with torch.no_grad():
        # placeholder trans seed
        trans = torch.zeros(1, 1, 3).to(device)

        latent_dict = model.inference(audio, speaker_id, motion_vq, masked_motion=None, mask=None)

        face_latent = latent_dict["rec_face"] if model.cfg.lf > 0 and model.cfg.cf == 0 else None
        upper_latent = latent_dict["rec_upper"] if model.cfg.lu > 0 and model.cfg.cu == 0 else None
        hands_latent = latent_dict["rec_hands"] if model.cfg.lh > 0 and model.cfg.ch == 0 else None
        lower_latent = latent_dict["rec_lower"] if model.cfg.ll > 0 and model.cfg.cl == 0 else None

        face_index = torch.max(F.log_softmax(latent_dict["cls_face"], dim=2), dim=2)[1] if model.cfg.cf > 0 else None
        upper_index = torch.max(F.log_softmax(latent_dict["cls_upper"], dim=2), dim=2)[1] if model.cfg.cu > 0 else None
        hands_index = torch.max(F.log_softmax(latent_dict["cls_hands"], dim=2), dim=2)[1] if model.cfg.ch > 0 else None
        lower_index = torch.max(F.log_softmax(latent_dict["cls_lower"], dim=2), dim=2)[1] if model.cfg.cl > 0 else None

        all_pred = motion_vq.decode(
            face_latent=face_latent,
            upper_latent=upper_latent,
            lower_latent=lower_latent,
            hands_latent=hands_latent,
            face_index=face_index,
            upper_index=upper_index,
            lower_index=lower_index,
            hands_index=hands_index,
            get_global_motion=True,
            ref_trans=trans[:, 0],
        )

    motion_pred = all_pred["motion_axis_angle"]  # (1, T, J*3) or similar
    t = motion_pred.shape[1]
    motion_pred = motion_pred.cpu().numpy().reshape(t, -1)

    face_pred = all_pred["expression"].cpu().numpy().reshape(t, -1)
    trans_pred = all_pred["trans"].cpu().numpy().reshape(t, -1)

    out_path = os.path.join(save_folder, f"{os.path.splitext(os.path.basename(audio_path))[0]}_output.npz")
    beat_format_save(out_path, motion_pred, upsample=30 // pose_fps, expressions=face_pred, trans=trans_pred)
    return t, out_path


def visualize_one(npz_path, audio_path, nopytorch3d=False):
    """
    Windows-safe visualization:
    - If nopytorch3d=True: we skip because the repo's render2d imports pytorch3d at import-time.
      (If you later implement a pytorch3d-free renderer, plug it here.)
    - If nopytorch3d=False: tries render2d (requires pytorch3d) and then fast_render if enabled.
    """
    if nopytorch3d:
        print("[info] --nopytorch3d set: skipping visualization (repo render2d requires pytorch3d on Windows).")
        return

    # Try 2D render (requires pytorch3d in this repo)
    try:
        motion_dict = np.load(npz_path, allow_pickle=True)
        from emage_utils.npz2pose import render2d

        v2d_face = render2d(motion_dict, (512, 512), face_only=True, remove_global=True)
        write_video(npz_path.replace(".npz", "_2dface.mp4"), v2d_face.permute(0, 2, 3, 1), fps=30)

        v2d_body = render2d(motion_dict, (720, 480), face_only=False, remove_global=True)
        write_video(npz_path.replace(".npz", "_2dbody.mp4"), v2d_body.permute(0, 2, 3, 1), fps=30)

        print("[info] 2D videos written:", npz_path.replace(".npz", "_2dface.mp4"), "and", npz_path.replace(".npz", "_2dbody.mp4"))
    except Exception as e:
        print("[warn] 2D visualization failed (likely missing pytorch3d/cv2). Error:", repr(e))

    # Try 3D render only if fast_render is enabled
    if fast_render is None:
        print("[info] fast_render disabled; skipping 3D render.")
        return

    try:
        fast_render.render_one_sequence_with_face(
            npz_path,
            os.path.dirname(npz_path),
            audio_path,
            model_folder="./emage_evaltools/smplx_models/",
        )
    except Exception as e:
        print("[warn] 3D visualization failed (OpenGL/EGL). Error:", repr(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_folder", type=str, default="./examples/audio")
    parser.add_argument("--save_folder", type=str, default="./examples/motion")
    parser.add_argument("--visualization", action="store_true")
    parser.add_argument("--nopytorch3d", action="store_true", help="Skip any visualization that requires pytorch3d.")
    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load motion VQ components
    face_motion_vq = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/face").to(device)
    upper_motion_vq = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/upper").to(device)
    lower_motion_vq = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/lower").to(device)
    hands_motion_vq = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/hands").to(device)
    global_motion_ae = EmageVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/global").to(device)

    motion_vq = EmageVQModel(
        face_model=face_motion_vq,
        upper_model=upper_motion_vq,
        lower_model=lower_motion_vq,
        hands_model=hands_motion_vq,
        global_model=global_motion_ae,
    ).to(device)
    motion_vq.eval()

    # Load EMAGE audio model
    model = EmageAudioModel.from_pretrained("H-Liu1997/emage_audio").to(device)
    model.eval()

    audio_files = [os.path.join(args.audio_folder, f) for f in os.listdir(args.audio_folder) if f.lower().endswith(".wav")]
    sr, pose_fps = model.cfg.audio_sr, model.cfg.pose_fps

    all_t = 0
    start_time = time.time()

    for audio_path in tqdm(audio_files, desc="Inference"):
        t, npz_path = inference(model, motion_vq, audio_path, device, args.save_folder, sr, pose_fps)
        all_t += t

        if args.visualization:
            visualize_one(npz_path, audio_path, args.nopytorch3d)

    print(f"Generated total {all_t / pose_fps:.2f} seconds of motion in {time.time() - start_time:.2f} seconds.")
    print("Outputs in:", os.path.abspath(args.save_folder))


if __name__ == "__main__":
    main()
