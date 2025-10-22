import torch, sys
print("python:", sys.executable)
print("torch.__version__:", getattr(torch, "__version__", None))
print("torch.version.cuda:", getattr(torch.version, "cuda", None))
print("torch.cuda.is_available():", torch.cuda.is_available())
try:
    if torch.cuda.is_available():
        print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch cuda probe failed:", e)