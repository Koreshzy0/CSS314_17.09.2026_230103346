import threading
import time

THREADS = 8
PER_THREAD = 1_000_000
TOTAL_OPS = THREADS * PER_THREAD

mutex_counter = 0
mutex_lock = threading.Lock()


def mutex_worker():
    global mutex_counter
    for _ in range(PER_THREAD):
        with mutex_lock:
            mutex_counter += 1


class AtomicCounter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def get(self):
        return self.value

    def compare_and_swap(self, expected, new):
        with self.lock:
            if self.value == expected:
                self.value = new
                return True
            return False


atomic_counter = AtomicCounter()
cas_failures = 0
cas_fail_lock = threading.Lock()


def cas_worker():
    global cas_failures
    local_fail = 0
    for _ in range(PER_THREAD):
        while True:
            current = atomic_counter.get()
            if atomic_counter.compare_and_swap(current, current + 1):
                break
            local_fail += 1
    with cas_fail_lock:
        cas_failures += local_fail


def benchmark(worker):
    threads = []
    start = time.perf_counter()

    for _ in range(THREADS):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return (time.perf_counter() - start) * 1000


mutex_counter = 0
mutex_time = benchmark(mutex_worker)
mutex_throughput = TOTAL_OPS / (mutex_time / 1000)

atomic_counter = AtomicCounter()
cas_failures = 0
cas_time = benchmark(cas_worker)
cas_throughput = TOTAL_OPS / (cas_time / 1000)

print("Primitive 1: Mutex (threading.Lock)")
print(f"Execution Time : {mutex_time:.3f} ms")
print(f"Throughput     : {mutex_throughput:,.0f} ops/sec")
print()

print("Primitive 2: Simulated CAS")
print(f"Execution Time : {cas_time:.3f} ms")
print(f"Throughput     : {cas_throughput:,.0f} ops/sec")
print(f"CAS Retries    : {cas_failures:,}")
print()

print(f"Throughput Ratio (CAS / Mutex): {cas_throughput / mutex_throughput:.2f}x")
print()
print(f"Final Mutex Counter : {mutex_counter:,}")
print(f"Final CAS Counter   : {atomic_counter.get():,}")