# app/cli.py
import argparse
import io
import json
import struct
from typing import Tuple

import numpy as np
import librosa
import soundfile as sf
import requests


DEFAULT_AUDIO = r"C:\soqqle\EMAGESoqq\examples\audio\Rachel.wav"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def audio_path_to_wav_bytes(path: str) -> bytes:
    """Load any audio file -> WAV bytes (PCM_16)."""
    y, sr = librosa.load(path, sr=None, mono=True)
    buf = io.BytesIO()
    sf.write(buf, y.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def decode_motion_blob(blob: bytes) -> Tuple[int, int, int, np.ndarray]:
    """
    Decode binary format:
      [u32 fps][u32 frames][u32 dims][float32 payload frames*dims]
    Returns: (fps, frames, dims, motion[F,D] float32)
    """
    if len(blob) < 12:
        raise ValueError(f"Blob too small ({len(blob)} bytes)")

    fps, frames, dims = struct.unpack("<III", blob[:12])
    expected_payload_bytes = frames * dims * 4
    expected_total = 12 + expected_payload_bytes

    if len(blob) != expected_total:
        raise ValueError(
            f"Unexpected blob size. got={len(blob)} expected={expected_total} "
            f"(fps={fps}, frames={frames}, dims={dims})"
        )

    motion = np.frombuffer(blob, dtype=np.float32, offset=12).reshape(frames, dims)
    return int(fps), int(frames), int(dims), motion


def post_octet_stream(url: str, wav_bytes: bytes, timeout: int = 600) -> requests.Response:
    return requests.post(
        url,
        data=wav_bytes,
        headers={"Content-Type": "application/octet-stream"},
        timeout=timeout,
    )


def print_bin_checks(blob: bytes, fps: int, frames: int, dims: int, motion: np.ndarray, head: int = 0, stats: bool = True):
    # Size check (again, but printed)
    expected_total = 12 + frames * dims * 4
    ok_size = (len(blob) == expected_total)

    # Finite check
    finite = bool(np.isfinite(motion).all())

    print(f"[verify] bytes={len(blob)} expected={expected_total} ok_size={ok_size}")
    print(f"[verify] shape={motion.shape} dtype={motion.dtype} finite={finite}")

    if stats:
        mn = float(np.min(motion))
        mx = float(np.max(motion))
        mean = float(np.mean(motion))
        std = float(np.std(motion))
        print(f"[stats] min={mn:.6f} max={mx:.6f} mean={mean:.6f} std={std:.6f}")

    if head and head > 0:
        flat = motion.ravel()
        n = min(int(head), flat.size)
        preview = flat[:n].tolist()
        print(f"[head] first_{n}_floats={preview}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", type=str, default=DEFAULT_AUDIO)
    p.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)

    p.add_argument("--mode", choices=["json", "raw"], default="json")
    p.add_argument("--endpoint", type=str, default=None, help="infer_bytes | infer_raw | infer")

    p.add_argument("--out_json", type=str, default=None)
    p.add_argument("--out_bin", type=str, default=None)
    p.add_argument("--out_npy", type=str, default=None)

    p.add_argument("--timeout", type=int, default=600)

    # NEW: validation / printing
    p.add_argument("--verify_bin", action="store_true", help="Validate + print checks for raw .bin response")
    p.add_argument("--print_stats", action="store_true", help="Print min/max/mean/std in raw mode")
    p.add_argument("--print_head", type=int, default=0, help="Print first N floats in raw mode")

    args = p.parse_args()

    wav_bytes = audio_path_to_wav_bytes(args.audio)

    if args.endpoint is None:
        endpoint = "infer_raw" if args.mode == "raw" else "infer_bytes"
    else:
        endpoint = args.endpoint.strip("/")

    url = f"{args.base_url.rstrip('/')}/{endpoint}"

    if args.mode == "json":
        r = post_octet_stream(url, wav_bytes, timeout=args.timeout)
        r.raise_for_status()

        data = r.json()
        print(json.dumps({"fps": data.get("fps"), "frames": data.get("frames"), "dims": data.get("dims")}, indent=2))

        if args.out_json:
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(data, f)
            print(f"Saved JSON to: {args.out_json}")

    else:
        r = post_octet_stream(url, wav_bytes, timeout=args.timeout)
        r.raise_for_status()

        blob = r.content
        fps, frames, dims, motion = decode_motion_blob(blob)

        print(json.dumps({"fps": fps, "frames": frames, "dims": dims}, indent=2))

        # NEW: verification output
        if args.verify_bin or args.print_stats or args.print_head:
            print_bin_checks(
                blob=blob,
                fps=fps,
                frames=frames,
                dims=dims,
                motion=motion,
                head=args.print_head,
                stats=(args.print_stats or args.verify_bin),
            )

        if args.out_bin:
            with open(args.out_bin, "wb") as f:
                f.write(blob)
            print(f"Saved RAW blob to: {args.out_bin}")

        if args.out_npy:
            np.save(args.out_npy, motion)
            print(f"Saved motion .npy to: {args.out_npy}")


if __name__ == "__main__":
    main()
