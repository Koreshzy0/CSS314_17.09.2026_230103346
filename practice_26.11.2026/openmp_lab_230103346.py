# SECTION 0: LOCAL OPENMP ENVIRONMENT & DETERMINISTIC SEED HARNESS
import os
os.environ["NUMBA_THREADING_LAYER"] = "omp"   # Force native OpenMP runtime (libgomp / iomp)
import time, hashlib, json, math
import numpy as np
import numba as nb
from numba import njit, prange, set_num_threads, get_num_threads, get_thread_id, parallel_chunksize

STUDENT_NAME = "Sultangazyyeva Altynay"   
STUDENT_ID = 230103346                    
assert isinstance(STUDENT_ID, int) and STUDENT_ID > 0, "Set a valid numeric STUDENT_ID!"
assert (os.cpu_count() or 1) >= 4, "Local multi-core CPU (>=4 logical cores) required; no Colab!"


# Trigger OpenMP layer initialization and verify backend
@njit(parallel=True)
def _omp_probe():
    acc = 0
    for i in prange(64):
        acc += get_thread_id() + 1
    return acc


_ = _omp_probe()
assert nb.threading_layer() == "omp", f"Expected 'omp' layer, got {nb.threading_layer()}"

rng = np.random.default_rng(STUDENT_ID)


N_T1 = 4_000_000 + (STUDENT_ID % 991) * 1024
BINS_T1 = 48 + (STUDENT_ID % 17)
ITERS_T2 = 35_000_000 + (STUDENT_ID % 503) * 10_000
PAD_STRIDE = 8 + (STUDENT_ID % 5) * 8        # >= 8 int64 slots (>= 64 bytes cache-line isolation)
ROWS_T3 = 12_000 + (STUDENT_ID % 257) * 16
SKEW_MOD = 37 + (STUDENT_ID % 19)
GRID_H = 384 + (STUDENT_ID % 4) * 32
GRID_W = 512 + (STUDENT_ID % 5) * 32
OMEGA_SOR = np.float64(1.72 + (STUDENT_ID % 11) * 0.015)

data_t1 = rng.integers(0, BINS_T1, size=N_T1, dtype=np.int32)
weights_t1 = rng.uniform(0.25, 2.50, size=N_T1).astype(np.float64)

print(f"Backend: {nb.threading_layer()} | Logical Cores: {os.cpu_count()} | Student: {STUDENT_ID}")

THREAD_COUNTS = [1, 2, 4, 8]
THREAD_COUNTS = [t for t in THREAD_COUNTS if t <= (os.cpu_count() or 1)]


def timed(fn, *args, repeats=3):
    """Best-of-N wall-clock time (seconds) and the last result."""
    best, res = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best, res


# =============================================================================
# TASK 1: RACE-CONDITION FORENSICS & THREAD-PRIVATE HISTOGRAM REDUCTION
# =============================================================================
@njit(parallel=True)
def buggy_hist_omp(indices, weights, num_bins):
    hist = np.zeros(num_bins, dtype=np.int64)
    w_sums = np.zeros(num_bins, dtype=np.float64)
    for i in prange(indices.shape[0]):
        b = indices[i]
        hist[b] += 1             # Data race across OpenMP threads
        w_sums[b] += weights[i]  # Non-associative floating-point race
    return hist, w_sums



FP_SCALE = 2.0 ** 36   # max total ~ 2.5 * 2^36 * 4.1e6 ~ 7e17 < 9.2e18 (no overflow)


@njit(parallel=True)
def safe_hist_omp(indices, weights, num_bins):
    max_threads = get_num_threads()
    priv_h = np.zeros((max_threads, num_bins), dtype=np.int64)
    priv_w = np.zeros((max_threads, num_bins), dtype=np.int64)
    worker_hits = np.zeros((max_threads, 8), dtype=np.int64)   # padded: one cache line per thread
    for i in prange(indices.shape[0]):
        tid = get_thread_id()
        b = indices[i]
        priv_h[tid, b] += 1
        priv_w[tid, b] += np.int64(weights[i] * FP_SCALE + 0.5)
        worker_hits[tid, 0] += 1
    
    hist = np.zeros(num_bins, dtype=np.int64)
    wfix = np.zeros(num_bins, dtype=np.int64)
    for t in range(max_threads):
        for b in range(num_bins):
            hist[b] += priv_h[t, b]
            wfix[b] += priv_w[t, b]
    w_sums = wfix.astype(np.float64) / FP_SCALE
    return hist, w_sums, worker_hits[:, 0].copy()


print("\n--- TASK 1 ---")

ref_hist = np.bincount(data_t1, minlength=BINS_T1).astype(np.int64)
ref_w = np.bincount(data_t1, weights=weights_t1, minlength=BINS_T1)


set_num_threads(min(8, os.cpu_count() or 1))
buggy_hist_omp(data_t1[:1000], weights_t1[:1000], BINS_T1)   
lost_counts, max_hist_diff, max_w_diff = [], 0, 0.0
for run in range(10):
    h, w = buggy_hist_omp(data_t1, weights_t1, BINS_T1)
    lost_counts.append(int(N_T1 - np.sum(h)))
    max_hist_diff = max(max_hist_diff, int(np.max(np.abs(h - ref_hist))))
    max_w_diff = max(max_w_diff, float(np.max(np.abs(w - ref_w))))
print(f"buggy lost counts per run : {lost_counts}")
print(f"max |hist - ref|          : {max_hist_diff}")
print(f"max |w_sums - ref|        : {max_w_diff:.6e}")


set_num_threads(1)
safe_hist_omp(data_t1[:1000], weights_t1[:1000], BINS_T1)    
set_num_threads(1)
base_h, base_w, _ = safe_hist_omp(data_t1, weights_t1, BINS_T1)
assert np.array_equal(base_h, ref_hist), "safe hist != reference"
assert np.max(np.abs(base_w - ref_w)) < 1e-3, "safe weights too far from reference"

for t in THREAD_COUNTS:
    set_num_threads(t)
    h, w, hits = safe_hist_omp(data_t1, weights_t1, BINS_T1)
    assert np.array_equal(h, base_h), f"hist differs at {t} threads"
    assert w.tobytes() == base_w.tobytes(), f"w_sums not bit-identical at {t} threads"
    assert int(np.sum(hits)) == N_T1, "worker_hits must add up to N_T1"
    print(f"safe @ {t} threads: bit-identical OK | active workers = "
          f"{int(np.count_nonzero(hits))}/{t} | worker_hits = {hits.tolist()}")

set_num_threads(min(8, os.cpu_count() or 1))
hist_t1, wsums_t1, _hits = safe_hist_omp(data_t1, weights_t1, BINS_T1)


DIAGNOSIS_T1 = (
    "Floating-point addition is not associative: (a+b)+c can differ from a+(b+c) because every "
    "intermediate sum is rounded to a 53-bit mantissa. In buggy_hist_omp the order in which threads "
    "add weights into w_sums[b] depends on scheduling, so the rounding errors accumulate differently "
    "from run to run. Even if the integer increments were atomic and no update were lost, the float "
    "totals could still change in the last bits between runs and between thread counts. In my runs the "
    "unsafe kernel lost about 2.4-2.7 million of 4,084,992 counts per run, and safe_hist_omp fixed this "
    "with one private row per thread, fixed-point int64 weight sums (exactly associative) and a merge "
    "in fixed order 0..max_threads-1, giving bit-identical output on 1, 2, 4 and 8 threads."
)


# =============================================================================
# TASK 2: ELIMINATING CACHE-LINE FALSE SHARING
# =============================================================================
@njit(parallel=True)
def unpadded_state_omp(iters, seed_val):
    n_threads = get_num_threads()
    state = np.zeros(n_threads, dtype=np.int64)
    for t in prange(n_threads):
        tid = get_thread_id()
        x = np.int64(seed_val + tid * 1315423911)
        for step in range(iters):
            x = (x * np.int64(6364136223846793005) + np.int64(1442695040888963407)) & np.int64(0x7FFFFFFFFFFFFFFF)
            state[tid] ^= (x >> 17)   # writes directly to a shared cache line on every iteration
    return state


@njit(parallel=True)
def padded_state_omp(iters, seed_val, pad_stride):
    n_threads = get_num_threads()
    padded_state = np.zeros((n_threads, pad_stride), dtype=np.int64)   # row = pad_stride*8 bytes
    for t in prange(n_threads):
        tid = get_thread_id()
        x = np.int64(seed_val + tid * 1315423911)
        for step in range(iters):
            x = (x * np.int64(6364136223846793005) + np.int64(1442695040888963407)) & np.int64(0x7FFFFFFFFFFFFFFF)
            padded_state[tid, 0] ^= (x >> 17)   # store stays inside the loop, own cache line
    return padded_state


print("\n--- TASK 2 ---")
set_num_threads(1)
unpadded_state_omp(1000, STUDENT_ID)                 # warm-up / compile
padded_state_omp(1000, STUDENT_ID, PAD_STRIDE)
print(f"PAD_STRIDE = {PAD_STRIDE} int64 slots = {PAD_STRIDE * 8} bytes")

t2_unpadded, t2_padded = {}, {}
for t in THREAD_COUNTS:
    set_num_threads(t)
    tu, su = timed(unpadded_state_omp, ITERS_T2, STUDENT_ID)
    tp, sp = timed(padded_state_omp, ITERS_T2, STUDENT_ID, PAD_STRIDE)
    assert np.array_equal(su, sp[:, 0]), f"padded != unpadded at {t} threads"
    t2_unpadded[t], t2_padded[t] = tu, tp
    print(f"{t} threads: unpadded {tu:.4f}s | padded {tp:.4f}s | ratio {tu / tp:.2f}x")

T_MAX = THREAD_COUNTS[-1]
speedup_t2 = t2_unpadded[T_MAX] / t2_padded[T_MAX]
print(f"Padding speedup at {T_MAX} threads: {speedup_t2:.2f}x")

set_num_threads(min(8, os.cpu_count() or 1))
state_t2 = unpadded_state_omp(ITERS_T2, STUDENT_ID)

DIAGNOSIS_T2 = (
    "In unpadded_state_omp the int64 slots of state[] sit next to each other inside one cache line "
    "(64 bytes on x86, 128 bytes on Apple M-series). Under the MESI protocol a core must hold the line "
    "in Modified or Exclusive state to write it, so each store invalidates that line in the L1 of the "
    "other cores, and the line ping-pongs between them even though no two threads share a variable. "
    "padded_state_omp places each thread's slot PAD_STRIDE*8 = 128 bytes apart, so each line has one "
    "owner and no invalidations are needed. In my measurements the effect was small (1.09x at 8 threads, "
    "1.00x at 1-4 threads) and the runtime stayed around 0.04-0.05 s for 35M iterations. I believe the "
    "LLVM optimizer kept state[tid] in a register inside the inner loop and wrote it to memory far less "
    "often than once per iteration, which hides most of the false sharing; the theory predicts a much "
    "larger slowdown if the store really reaches the cache on every step."
)



# =============================================================================
# TASK 3: IRREGULAR WORKLOAD SKEW & DYNAMIC CHUNK SCHEDULING
# =============================================================================
work_units = np.array(
    [40 + ((r * 7919 + STUDENT_ID) % SKEW_MOD) ** 2 for r in range(ROWS_T3)], dtype=np.int64
)


@njit
def row_score(r, units):
    """Heavy trigonometric-polynomial kernel for one row (identical code for both schedulers)."""
    s = 0.0
    base = 0.001 * (r % 1000)
    for k in range(units):
        x = base + 0.0137 * k
        s += math.sin(x) * math.cos(x) + math.sqrt(abs(x) + 1.0)
    return s


@njit(parallel=True)
def static_skew_omp(units):
    n_rows = units.shape[0]
    n_threads = get_num_threads()
    out = np.zeros(n_rows, dtype=np.float64)
    ops = np.zeros((n_threads, 8), dtype=np.int64)      
    for r in prange(n_rows):                            
        tid = get_thread_id()
        out[r] = row_score(r, units[r])
        ops[tid, 0] += units[r]
    return out, ops[:, 0].copy()


@njit(parallel=True)
def balanced_skew_omp(units, chunk):
    n_rows = units.shape[0]
    n_threads = get_num_threads()
    out = np.zeros(n_rows, dtype=np.float64)
    ops = np.zeros((n_threads, 8), dtype=np.int64)
    # Chunked scheduling: the row range is cut into small chunks (e.g. 16 rows) that
    # worker threads claim one after another until all rows are exhausted.
    with parallel_chunksize(chunk):
        for r in prange(n_rows):
            tid = get_thread_id()
            out[r] = row_score(r, units[r])
            ops[tid, 0] += units[r]
    return out, ops[:, 0].copy()


def imbalance(ops):
    active = ops[ops > 0]
    return float(active.max() / active.min()) if active.size else float("inf")


print("\n--- TASK 3 ---")
CHUNK = 16
set_num_threads(1)
static_skew_omp(work_units[:64].copy())
balanced_skew_omp(work_units[:64].copy(), CHUNK)

for t in THREAD_COUNTS:
    set_num_threads(t)
    ts, (o_s, ops_s) = timed(static_skew_omp, work_units)
    tb, (o_b, ops_b) = timed(balanced_skew_omp, work_units, CHUNK)
    print(f"{t} threads: static {ts:.4f}s (LIR {imbalance(ops_s):.3f}) | "
          f"balanced {tb:.4f}s (LIR {imbalance(ops_b):.3f})")

set_num_threads(min(8, os.cpu_count() or 1))
ts8, (out_static, ops_static) = timed(static_skew_omp, work_units)
tb8, (out_balanced, ops_balanced) = timed(balanced_skew_omp, work_units, CHUNK)
out_t3 = out_balanced

lir_static, lir_bal = imbalance(ops_static), imbalance(ops_balanced)
print(f"static   ops/thread = {ops_static.tolist()}  LIR = {lir_static:.3f}  time = {ts8:.4f}s")
print(f"balanced ops/thread = {ops_balanced.tolist()}  LIR = {lir_bal:.3f}  time = {tb8:.4f}s")

max_err = float(np.max(np.abs(out_static - out_balanced)))
assert max_err < 1e-11, f"static/balanced mismatch: {max_err}"
if lir_bal < 1.15:
    print(f"Validation gate PASSED: max|diff| = {max_err:.2e}, LIR = {lir_bal:.3f} < 1.15")
else:
    print(f"WARNING: LIR {lir_bal:.3f} >= 1.15 (see note about P/E cores on Apple Silicon)")

DIAGNOSIS_T3 = (
    "Numba's default prange splits the row range into one contiguous block per thread and ends with an "
    "implicit barrier, so the loop waits for the thread with the most expensive block. Here the row cost "
    "40 + (skew)^2 is pseudo-random but periodic in r, so every contiguous block of about 1500 rows "
    "contains many full periods, and the static schedule is already almost perfectly balanced: all eight "
    "threads executed exactly 1,309,140 operations (Load Imbalance Ratio 1.000). balanced_skew_omp cuts "
    "the rows into chunks of 16 via parallel_chunksize so that threads can take new chunks as they "
    "finish; it gave LIR 1.009 (below the 1.15 gate) and the same results as the static kernel "
    "(max difference 0.0), but no speedup (0.0158 s vs 0.0151 s at 8 threads). Dynamic chunking pays "
    "off only when the static blocks are really unequal, and its scheduling overhead is visible when "
    "they are not. Per-thread operation counters are padded to a full cache line to avoid false sharing."
)


# =============================================================================
# TASK 4: PARALLEL RED-BLACK SOR PDE SOLVER
# =============================================================================
def make_grid():
    U = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    U[0, :] = 100.0
    U[-1, :] = -50.0
    U[:, 0] = 25.0 * np.sin(np.linspace(0, np.pi, GRID_H))
    U[:, -1] = 75.0 * np.cos(np.linspace(0, np.pi / 2, GRID_H))
    return U


@njit(parallel=True)
def sor_red_black_step(U, omega, color, deltas):
    """One colour sweep. color=0 -> Red ((r+c)%2==0), color=1 -> Black ((r+c)%2==1)."""
    H, W = U.shape
    for r in prange(1, H - 1):
        tid = get_thread_id()
        local = 0.0
        c0 = 1 if ((r + 1) % 2) == color else 2     
        for c in range(c0, W - 1, 2):
            new = (1.0 - omega) * U[r, c] + 0.25 * omega * (
                U[r - 1, c] + U[r + 1, c] + U[r, c - 1] + U[r, c + 1]
            )
            d = abs(new - U[r, c])
            if d > local:
                local = d
            U[r, c] = new
        if local > deltas[tid, 0]:                  
            deltas[tid, 0] = local


@njit
def solve_pde_omp(U0, omega, sweeps):
    U = U0.copy()
    n_threads = get_num_threads()
    deltas = np.zeros((n_threads, 8), dtype=np.float64)   
    max_delta = 0.0
    for it in range(sweeps):
        deltas[:, :] = 0.0
        sor_red_black_step(U, omega, 0, deltas)   
        sor_red_black_step(U, omega, 1, deltas)   
        max_delta = 0.0
        for t in range(n_threads):                
            if deltas[t, 0] > max_delta:
                max_delta = deltas[t, 0]
    return U, max_delta


print("\n--- TASK 4 ---")
SWEEPS = 600
U_init = make_grid()
set_num_threads(1)
solve_pde_omp(U_init, OMEGA_SOR, 2)                   

set_num_threads(1)
t_ref, (grid_ref, delta_ref) = timed(solve_pde_omp, U_init, OMEGA_SOR, SWEEPS, repeats=1)
print(f"grid {GRID_H}x{GRID_W}, omega = {float(OMEGA_SOR):.3f}, sweeps = {SWEEPS}")
print(f"1 thread : {t_ref:.4f}s | speedup 1.00x | max_delta = {delta_ref:.3e}")

grid_t4 = grid_ref
for t in THREAD_COUNTS:
    if t == 1:
        continue
    set_num_threads(t)
    tt, (g, dlt) = timed(solve_pde_omp, U_init, OMEGA_SOR, SWEEPS, repeats=1)
    max_abs_diff = float(np.max(np.abs(g - grid_ref)))
    assert max_abs_diff < 1e-9, f"{t} threads differ from 1-thread run by {max_abs_diff}"
    print(f"{t} threads: {tt:.4f}s | speedup {t_ref / tt:.2f}x | max_delta = {dlt:.3e} | "
          f"max_abs_diff = {max_abs_diff:.2e}")
    if t == min(8, os.cpu_count() or 1):
        grid_t4 = g


# =============================================================================
# FINAL SECTION: DETERMINISTIC VERIFICATION & TELEMETRY BUNDLE
# =============================================================================
set_num_threads(8 if (os.cpu_count() or 1) >= 8 else os.cpu_count())
for idx, diag in enumerate([DIAGNOSIS_T1, DIAGNOSIS_T2, DIAGNOSIS_T3], start=1):
    assert isinstance(diag, str) and len(diag.strip()) >= 80, f"Complete DIAGNOSIS_T{idx}!"

checksum_blob = (
    f"{STUDENT_ID}|"
    f"{int(np.sum(hist_t1))}|{float(np.sum(wsums_t1)):.4f}|"
    f"{int(np.bitwise_xor.reduce(state_t2))}|"
    f"{float(np.sum(out_t3)):.6f}|"
    f"{float(np.sum(grid_t4)):.6f}"
)
lab_digest = hashlib.sha256(checksum_blob.encode()).hexdigest()[:20].upper()

bundle = {
    "instructor": "Instructor - Sufyan bin Uzayr",
    "student_name": STUDENT_NAME,
    "student_id": STUDENT_ID,
    "omp_layer": nb.threading_layer(),
    "cpu_cores": os.cpu_count(),
    "verification_digest": lab_digest,
}
print("\n=== OPENMP PYTHON LAB VERIFICATION BUNDLE ===")
print(json.dumps(bundle, indent=2))
