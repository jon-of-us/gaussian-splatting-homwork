"""Check the chosen CUDA environment without imposing a machine-specific version."""
from pathlib import Path
import argparse
from io import BytesIO
import re
import subprocess
import sys
import warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-runtime", action="store_true",
                        help="Also check NumPy/Torch conversion, CUDA arithmetic and image saving after dependency installation")
    args = parser.parse_args()
    try:
        import torch
        import torchvision
        from torch.utils.cpp_extension import CUDA_HOME
    except (ImportError, RuntimeError, OSError) as error:
        raise SystemExit(
            "Install a compatible CUDA-enabled PyTorch/torchvision pair in the active "
            "environment using https://pytorch.org/get-started/locally/ before setup. "
            f"Import failed: {error}"
        ) from error
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise SystemExit("A CUDA-enabled PyTorch build, supported NVIDIA GPU and working driver are required.")
    if CUDA_HOME is None:
        raise SystemExit("CUDA Toolkit was not found; install it and set CUDA_HOME to its directory.")
    nvcc = Path(CUDA_HOME) / "bin" / ("nvcc.exe" if sys.platform == "win32" else "nvcc")
    if not nvcc.is_file():
        raise SystemExit(f"CUDA Toolkit compiler not found: {nvcc}")
    toolkit = re.search(r"release (\d+)\.(\d+)", subprocess.check_output([str(nvcc), "--version"], text=True))
    torch_cuda = re.match(r"(\d+)\.(\d+)", torch.version.cuda)
    if toolkit is None or torch_cuda is None:
        raise SystemExit("Could not determine Toolkit/PyTorch CUDA versions; check the selected installation.")
    toolkit_version = tuple(map(int, toolkit.groups()))
    torch_version = tuple(map(int, torch_cuda.groups()))
    if toolkit_version[0] != torch_version[0]:
        raise SystemExit("CUDA Toolkit and PyTorch CUDA major versions differ; choose a compatible combination.")
    if toolkit_version != torch_version:
        warnings.warn(
            "Toolkit and PyTorch CUDA minor versions differ. Prefer matching versions; "
            "this combination still needs to pass extension compilation and CUDA validation.",
            RuntimeWarning,
        )
    if args.check_runtime:
        try:
            import numpy as np
            values = np.arange(8, dtype=np.float32)
            tensor = torch.from_numpy(values.copy())
            if not np.array_equal(tensor.numpy(), values):
                raise RuntimeError("CPU tensor/NumPy round trip changed values")
            result = (tensor.cuda() * 2 + 1).cpu().numpy()
            if not np.array_equal(result, values * 2 + 1):
                raise RuntimeError("CUDA arithmetic or NumPy transfer produced incorrect values")
            image_buffer = BytesIO()
            torchvision.utils.save_image(torch.zeros(3, 2, 2), image_buffer, format="PNG")
            if not image_buffer.getvalue().startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeError("Could not save a PNG through torchvision")
        except Exception as error:
            raise SystemExit(
                "Runtime interoperability check failed. Choose compatible PyTorch, torchvision, "
                f"NumPy, GPU and driver builds. Details: {error}"
            ) from error
        print("NumPy/Torch conversion, CUDA arithmetic and PNG saving: passed")
    print("Active Python:", sys.executable)
    print("PyTorch:", torch.__version__, "torchvision:", torchvision.__version__)
    print("PyTorch CUDA:", torch.version.cuda, "Toolkit:", ".".join(map(str, toolkit_version)))
    print("Environment preflight passed; compiler compatibility is checked during the build.")


if __name__ == "__main__":
    main()
