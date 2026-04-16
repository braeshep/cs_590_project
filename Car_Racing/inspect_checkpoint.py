"""
inspect_checkpoint.py
Peek inside agent_weights.pt to see its structure so we can load it correctly.
"""
import torch

ckpt = torch.load(r"weights\agent_weights.pt", map_location="cpu")

print("Type:", type(ckpt))

if isinstance(ckpt, dict):
    print("Keys:", list(ckpt.keys()))
    for k, v in ckpt.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")
        elif isinstance(v, dict):
            print(f"  {k}: (dict with keys {list(v.keys())[:5]})")
        else:
            print(f"  {k}: {type(v)}")
else:
    print("Not a dict — it's a raw object:", type(ckpt))
    if hasattr(ckpt, 'state_dict'):
        print("Has state_dict(), keys:", list(ckpt.state_dict().keys())[:10])
