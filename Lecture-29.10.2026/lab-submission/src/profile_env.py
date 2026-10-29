import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import psutil


def command_output(command):
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def get_cpu_info():
    system = platform.system()
    model = platform.processor() or "Unknown"
    microarchitecture = "Unknown"
    base_frequency_mhz = None

    if system == "Linux":
        try:
            info = Path("/proc/cpuinfo").read_text()
            for line in info.splitlines():
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
            for line in info.splitlines():
                if line.startswith("cpu MHz"):
                    base_frequency_mhz = float(
                        line.split(":", 1)[1].strip()
                    )
                    break
        except OSError:
            pass
        microarchitecture = command_output(["lscpu"]) or "Unknown"

    elif system == "Darwin":
        model = command_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"]
        ) or model
        frequency = command_output(
            ["sysctl", "-n", "hw.cpufrequency"]
        )
        if frequency and frequency.isdigit():
            base_frequency_mhz = int(frequency) / 1_000_000

    return {
        "model": model,
        "microarchitecture_details": microarchitecture,
        "base_frequency_mhz": base_frequency_mhz,
    }


def cache_info():
    result = {
        "L1_data_bytes": None,
        "L2_bytes": None,
        "L3_bytes": None,
        "cache_line_bytes": 64,
    }

    if platform.system() == "Linux":
        base = Path("/sys/devices/system/cpu/cpu0/cache")
        if base.exists():
            for index in base.glob("index*"):
                try:
                    level = (index / "level").read_text().strip()
                    kind = (index / "type").read_text().strip()
                    size_text = (index / "size").read_text().strip()
                    line_size = int(
                        (index / "coherency_line_size").read_text().strip()
                    )
                    size = int(
                        size_text.upper().replace("K", "").replace("M", "")
                    )
                    if size_text.upper().endswith("K"):
                        size *= 1024
                    elif size_text.upper().endswith("M"):
                        size *= 1024 * 1024

                    if level == "1" and kind == "Data":
                        result["L1_data_bytes"] = size
                    elif level == "2":
                        result["L2_bytes"] = size
                    elif level == "3":
                        result["L3_bytes"] = size
                    result["cache_line_bytes"] = line_size
                except (OSError, ValueError):
                    continue

    elif platform.system() == "Darwin":
        for key, name in [
            ("L1_data_bytes", "hw.l1dcachesize"),
            ("L2_bytes", "hw.l2cachesize"),
            ("L3_bytes", "hw.l3cachesize"),
        ]:
            value = command_output(["sysctl", "-n", name])
            if value and value.isdigit():
                result[key] = int(value)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from salt import get_student_params

    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)

    profile = {
        "cpu": get_cpu_info(),
        "physical_cores": physical,
        "logical_threads": logical,
        "smt_enabled": logical > physical if physical else None,
        "cache": cache_info(),
        "os": platform.system(),
        "os_release": platform.release(),
        "kernel_version": platform.version(),
        "machine": platform.machine(),
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
    }

    results = Path("results")
    results.mkdir(exist_ok=True)

    (results / "host_profile.json").write_text(
        json.dumps(profile, indent=2)
    )
    (results / "salted_params.json").write_text(
        json.dumps(get_student_params(args.student_id), indent=2)
    )

    print(json.dumps(profile, indent=2))
    print("Saved host_profile.json and salted_params.json")


if __name__ == "__main__":
    main()
