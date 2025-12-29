import sys
import numpy as np

p = sys.argv[1]
d = np.load(p, allow_pickle=True)
print("FILE:", p)
print("KEYS:", d.files)
for k in d.files:
    v = d[k]
    try:
        print(f"{k}: shape={v.shape}, dtype={v.dtype}")
    except Exception:
        print(f"{k}: type={type(v)}")

#python inspect_npz.py .\examples\motion\2_scott_0_103_103_28s_output.npz 