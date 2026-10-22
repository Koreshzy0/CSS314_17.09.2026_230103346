import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import platform
import subprocess
import time

import numpy as np

STUDENT_ID = 230103346          
N = 20_000_000


def cpu_model() -> str:
    """Best-effort CPU model string across Windows / Linux / macOS."""
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "Unknown"


def main() -> None:
    print(f"Student ID      : {STUDENT_ID}")
    print(f"OS              : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Platform        : {platform.platform()}")
    print(f"CPU model       : {cpu_model()}")
    print(f"Logical cores   : {os.cpu_count()}")
    print(f"Python / NumPy  : {platform.python_version()} / {np.__version__}")

  
    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(N, dtype=np.float64)
    print(f"Array           : {data.size:,} elements, dtype={data.dtype}")

  
    start = time.perf_counter()
    total = float(np.sum(data))
    std = float(np.std(data))
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print(f"Sum             : {total:.6f}")
    print(f"Std deviation   : {std:.10f}")
    print(f"Execution time  : {elapsed_ms:.3f} ms")


if __name__ == "__main__":
    main()