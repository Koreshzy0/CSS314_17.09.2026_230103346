import platform
import queue
import threading
import time

STUDENT_ID = 230103346
ITEMS = 1000
WORKERS = 4
UNIT = 0.0001


def cost(i: int) -> float:
    return UNIT * (i % 20)


def do_work(i: int) -> None:
    time.sleep(cost(i))


def static_run(workers: int):
    block = ITEMS // workers
    finish = [0.0] * workers
    start_evt = threading.Event()
    t0 = 0.0

    def worker(w: int) -> None:
        start_evt.wait()
        for i in range(w * block, (w + 1) * block):
            do_work(i)
        finish[w] = time.perf_counter() - t0

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(workers)]
    for t in threads:
        t.start()
    t0 = time.perf_counter()
    start_evt.set()
    for t in threads:
        t.join()
    total = time.perf_counter() - t0
    return total, finish


def dynamic_run(workers: int, chunk: int):
    q = queue.Queue()
    for s in range(0, ITEMS, chunk):
        q.put(range(s, min(s + chunk, ITEMS)))
    finish = [0.0] * workers
    start_evt = threading.Event()
    t0 = 0.0

    def worker(w: int) -> None:
        start_evt.wait()
        while True:
            try:
                work = q.get_nowait()
            except queue.Empty:
                break
            for i in work:
                do_work(i)
        finish[w] = time.perf_counter() - t0

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(workers)]
    for t in threads:
        t.start()
    t0 = time.perf_counter()
    start_evt.set()
    for t in threads:
        t.join()
    total = time.perf_counter() - t0
    return total, finish


def report(name: str, total: float, finish: list) -> None:
    delta = (max(finish) - min(finish)) * 1000.0
    print(f"{name:<28} | {total * 1000:>10.3f} | {min(finish) * 1000:>10.3f} | "
          f"{max(finish) * 1000:>10.3f} | {delta:>10.3f}")


def main() -> None:
    total_work_ms = sum(cost(i) for i in range(ITEMS)) * 1000.0

    print(f"Student ID    : {STUDENT_ID}")
    print(f"OS            : {platform.system()} {platform.release()}")
    print(f"Items         : {ITEMS}, cost(i) = sleep(0.0001 * (i % 20))")
    print(f"Workers       : {WORKERS}")
    print(f"Total work    : {total_work_ms:.3f} ms")
    print(f"Ideal time    : {total_work_ms / WORKERS:.3f} ms (perfect balance, zero overhead)")
    print()

    header = (f"{'Strategy':<28} | {'Total (ms)':>10} | {'First (ms)':>10} | "
              f"{'Last (ms)':>10} | {'Delta (ms)':>10}")
    print(header)
    print("-" * len(header))

    total, finish = static_run(WORKERS)
    report("A: Static blocks (250 each)", total, finish)

    for chunk in (5, 1):
        total, finish = dynamic_run(WORKERS, chunk)
        report(f"B: Dynamic queue (chunk={chunk})", total, finish)

    print()
    print("Delta = time between the first and the last worker finishing.")


if __name__ == "__main__":
    main()
