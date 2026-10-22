import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import time
from multiprocessing import shared_memory

import numpy as np

STUDENT_ID = 230103346
SIZES = [100_000, 1_000_000, 10_000_000]


def cpu_model() -> str:
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


def attach(name: str) -> shared_memory.SharedMemory:
    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:
        shm = shared_memory.SharedMemory(name=name)
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
        return shm


def worker(conn) -> None:
    while True:
        msg = conn.recv()
        if isinstance(msg, str) and msg == "stop":
            break
        if isinstance(msg, tuple):
            _, name, n = msg
            shm = attach(name)
            view = np.ndarray((n,), dtype=np.float64, buffer=shm.buf)
            reply = (float(view[0]), float(view[-1]))
            del view
            shm.close()
        else:
            reply = (float(msg[0]), float(msg[-1]))
        conn.send(reply)
    conn.close()


def reps_for(n: int) -> int:
    if n <= 100_000:
        return 50
    if n <= 1_000_000:
        return 20
    return 7


def median_ms(fn, reps: int) -> float:
    fn()
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000.0)
    return statistics.median(times)


def main() -> None:
    print(f"Student ID    : {STUDENT_ID}")
    print(f"OS / CPU      : {platform.system()} {platform.release()} / {cpu_model()}")
    print(f"Logical cores : {os.cpu_count()}")
    print(f"Python        : {platform.python_version()}, start method = {mp.get_start_method()}")
    print("Mode          : CPU only (no GPU): parent process -> child process, float64 arrays")
    print("Path A        : Pipe + pickle serialization (copies the whole array)")
    print("Path B        : multiprocessing.shared_memory (child maps the same pages, zero-copy)")
    print("Timing        : median of repeated runs; overhead = send until child's acknowledgement")
    print()

    parent_conn, child_conn = mp.Pipe(duplex=True)
    proc = mp.Process(target=worker, args=(child_conn,), daemon=True)
    proc.start()

    header = (f"{'N':>12} | {'Size (MB)':>9} | {'Path A: pickle (ms)':>19} | "
              f"{'Path B: shm (ms)':>16} | {'B incl. fill (ms)':>17} | {'A / B':>8}")
    print(header)
    print("-" * len(header))

    rng = np.random.default_rng(STUDENT_ID)
    try:
        for n in SIZES:
            data = rng.random(n, dtype=np.float64)
            expected = (float(data[0]), float(data[-1]))
            reps = reps_for(n)

            def path_a() -> None:
                parent_conn.send(data)
                reply = parent_conn.recv()
                if reply != expected:
                    raise RuntimeError("Path A: data mismatch")

            shm = shared_memory.SharedMemory(create=True, size=n * 8)
            try:
                shm_view = np.ndarray((n,), dtype=np.float64, buffer=shm.buf)
                shm_view[:] = data

                def path_b() -> None:
                    parent_conn.send(("shm", shm.name, n))
                    reply = parent_conn.recv()
                    if reply != expected:
                        raise RuntimeError("Path B: data mismatch")

                def fill() -> None:
                    shm_view[:] = data

                t_a = median_ms(path_a, reps)
                t_b = median_ms(path_b, reps)
                t_fill = median_ms(fill, reps)
            finally:
                del shm_view
                shm.close()
                shm.unlink()

            print(f"{n:>12,} | {n * 8 / 1e6:>9.1f} | {t_a:>19.3f} | {t_b:>16.3f} | "
                  f"{t_b + t_fill:>17.3f} | {t_a / t_b:>7.1f}x")
    finally:
        parent_conn.send("stop")
        proc.join(timeout=5)

    print()
    print("Path B = handoff of data that already lives in shared memory (only the segment name is sent).")
    print("B incl. fill = Path B plus one memcpy of the array into the shared segment.")


if __name__ == "__main__":
    main()
