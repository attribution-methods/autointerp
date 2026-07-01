import traceback

import torch

print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))

try:
    a = torch.randn(128, 128, device="cuda")
    b = torch.randn(128, 128, device="cuda")
    print("allocated")
    c = a @ b
    torch.cuda.synchronize()
    print("matmul ok", float(c[0, 0]))
except Exception:
    traceback.print_exc()
    raise
