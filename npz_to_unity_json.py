import sys, json
import numpy as np

# --- These masks come from your EmageVQModel code snippet ---
JOINT_MASK_UPPER = [
  False, False, False, True, False, False, True, False, False, True,
  False, False, True, True, True, True, True, True, True, True,
  True, True, False, False, False, False, False, False, False, False,
  False, False, False, False, False, False, False, False, False, False,
  False, False, False, False, False, False, False, False, False, False,
  False, False, False, False, False
]

JOINT_MASK_LOWER = [
  True, True, True, False, True, True, False, True, True, False,
  True, True, False, False, False, False, False, False, False, False,
  False, False, False, False, False, False, False, False, False, False,
  False, False, False, False, False, False, False, False, False, False,
  False, False, False, False, False, False, False, False, False, False,
  False, False, False, False, False
]

# Hands are explicitly 25..54 in the model code
HANDS_RANGE = list(range(25, 55))
JAW_INDEX = 22

def peek(name, arr, n=2, decimals=4):
    """Print shape + first n rows (truncated) for quick sanity checks."""
    if arr is None:
        print(f"[{name}] None")
        return
    a = np.asarray(arr)
    print(f"[{name}] shape={a.shape} dtype={a.dtype}")
    if a.ndim == 1:
        show = a[: min(len(a), n*10)]
        print(f"  head: {np.round(show, decimals)}")
    elif a.ndim >= 2:
        rows = min(a.shape[0], n)
        cols = min(a.shape[1], 12)
        print(f"  head[{rows}x{cols}]:\n{np.round(a[:rows, :cols], decimals)}")


def idx_from_mask(mask):
    return [i for i, v in enumerate(mask) if v]

def slice_joints_axis_angle(poses_F165: np.ndarray, joint_indices):
    """
    poses_F165 is axis-angle (F,165) where joint j occupies [j*3 : j*3+3].
    Returns (F, len(joint_indices)*3) packed in the given order.
    """
    F = poses_F165.shape[0]
    out = np.zeros((F, len(joint_indices) * 3), dtype=np.float32)
    for k, j in enumerate(joint_indices):
        out[:, k*3:(k+1)*3] = poses_F165[:, j*3:(j+1)*3]
    return out

def main():
    inp = sys.argv[1]
    out = sys.argv[2]

    d = np.load(inp, allow_pickle=True)
    print("[npz] keys:", d.files)

    if "poses" in d: peek("poses", d["poses"])
    if "trans" in d: peek("trans", d["trans"])
    if "expressions" in d: peek("expressions", d["expressions"])
    if "mocap_frame_rate" in d: print("[npz] mocap_frame_rate:", d["mocap_frame_rate"])
    poses = d["poses"].astype(np.float32)   # (F,165) axis-angle
    trans = d["trans"].astype(np.float32)   # (F,3)

    fps = int(d["mocap_frame_rate"]) if "mocap_frame_rate" in d else 30
    frames = int(poses.shape[0])

    upper_idx = idx_from_mask(JOINT_MASK_UPPER)
    lower_idx = idx_from_mask(JOINT_MASK_LOWER)
    hands_idx = HANDS_RANGE
    jaw_idx = [JAW_INDEX]

    upper = slice_joints_axis_angle(poses, upper_idx)
    lower = slice_joints_axis_angle(poses, lower_idx)
    hands = slice_joints_axis_angle(poses, hands_idx)
    jaw = slice_joints_axis_angle(poses, jaw_idx)  # (F,3)
    
    print("[idx] upper:", len(upper_idx), "first:", upper_idx[:10])
    print("[idx] lower:", len(lower_idx), "first:", lower_idx[:10])
    print("[idx] hands:", len(hands_idx), "range:", hands_idx[0], "..", hands_idx[-1])
    print("[idx] jaw:", jaw_idx)

    # (Optional) sanity checks
    assert poses.shape[1] == 165, f"Expected pose_dims=165, got {poses.shape[1]}"
    assert len(upper_idx) > 0 and len(lower_idx) > 0, "Upper/lower masks produced empty sets."
    assert jaw.shape == (frames, 3), f"Jaw should be (F,3), got {jaw.shape}"

    obj = {
        "fps": fps,
        "frames": frames,

        # root translation (SMPL-X global translation, in meters usually)
        "trans_flat": trans.reshape(-1).tolist(),

        # split pose payloads
        "upper": {
            "joint_indices": upper_idx,
            "aa_flat": upper.reshape(-1).tolist()
        },
        "lower": {
            "joint_indices": lower_idx,
            "aa_flat": lower.reshape(-1).tolist()
        },
        "hands": {
            "joint_indices": hands_idx,
            "aa_flat": hands.reshape(-1).tolist()
        },
        "jaw": {
            "joint_indices": jaw_idx,
            "aa_flat": jaw.reshape(-1).tolist()
        }
    }

    # expressions exist in your NPZ but are optional for Unity right now
    if "expressions" in d:
        expr = d["expressions"].astype(np.float32)  # (F,100)
        obj["expressions"] = {
            "dims": int(expr.shape[1]),
            "flat": expr.reshape(-1).tolist()
        }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f)

    print("Wrote:", out)
    print("fps:", fps, "frames:", frames)
    print("upper joints:", len(upper_idx), "lower joints:", len(lower_idx), "hands joints:", len(hands_idx), "jaw:", jaw_idx)

if __name__ == "__main__":
    main()

# installation
#python -m venv .venv
#.\.venv\Scripts\Activate.ps1
#python -m pip install --upgrade pip
#python -m pip install numpy

#run example
#python npz_to_unity_json.py .\examples\motion\2_scott_0_103_103_28s_output.npz .\examples\motion\emage_unity.json