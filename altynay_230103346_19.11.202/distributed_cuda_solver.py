import os
import sys
import json
import math
import re
import time
import hashlib
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, float64

# Compile-time constants for 2D shared-memory tiling (16x16 interior + 1-cell halo apron)
TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
LAMBDA_DAMP = 0.15
REPORT_EVERY = 100
DT_SAFETY = 0.5  # dt = DT_SAFETY * dt_max

DIRS = ("up", "down", "left", "right")
OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}
TAG = {"up": 0, "down": 1, "left": 2, "right": 3}


def get_gpu_uuid() -> str:
    """Query active NVIDIA GPU UUID via nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    """Derive deterministic 64-bit integer seed bound to student GitHub handle and GPU UUID."""
    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}".encode("utf8")).hexdigest()
    return int(digest[:16], 16)


def allreduce_scalar(cart, value, op):
    s = np.array([value], dtype=np.float64)
    r = np.empty(1, dtype=np.float64)
    cart.Allreduce(s, r, op=op)
    return float(r[0])


# ==============================================================================
# TASK 1: Distributed Cartesian Topology & Hardware Context Initialization
# ==============================================================================
def init_distributed_context(github_user: str):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks (2x2 grid), got {size}.")

    cart = comm.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
    rank = cart.Get_rank()
    row_rank, col_rank = cart.Get_coords(rank)
    nbr_up, nbr_down = cart.Shift(0, 1)      # axis 0: (source=up, dest=down)
    nbr_left, nbr_right = cart.Shift(1, 1)   # axis 1: (source=left, dest=right)

    # Bind process to GPU 0 on Google Colab
    cuda.select_device(0)
    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = cart.bcast(gpu_uuid, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)

    return {
        "cart": cart,
        "rank": rank,
        "size": size,
        "row_rank": row_rank,
        "col_rank": col_rank,
        "nbr": {"up": nbr_up, "down": nbr_down, "left": nbr_left, "right": nbr_right},
        "gpu_uuid": gpu_uuid,
        "seed": instance_seed,
        "github_user": github_user.strip(),
        "ny": GLOBAL_NY // 2,
        "nx": GLOBAL_NX // 2,
    }


# ==============================================================================
# TASK 2: Seeded Heterogeneous Subdomain Initialization
# ==============================================================================
def initialize_subdomains(ctx):
    ny, nx = ctx["ny"], ctx["nx"]
    dx = 1.0 / (GLOBAL_NX + 1)
    # global node indices 0..513 (0 and 513 are the physical Dirichlet boundary)
    gi = ctx["row_rank"] * ny + np.arange(ny + 2)
    gj = ctx["col_rank"] * nx + np.arange(nx + 2)
    X, Y = np.meshgrid(gj * dx, gi * dx)  # X[i,j] = x[j], Y[i,j] = y[i]

    # heterogeneous diffusivity in [0.08, 0.24]
    kappa = 0.16 + 0.08 * np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)

    # seed-derived pulses (every rank draws the same sequence -> consistent global field)
    rng = np.random.default_rng(ctx["seed"])
    n_p = 3
    cx = rng.uniform(0.25, 0.75, n_p)
    cy = rng.uniform(0.25, 0.75, n_p)
    amp = rng.uniform(0.5, 1.0, n_p)
    sig = rng.uniform(0.05, 0.10, n_p)
    sx, sy = rng.uniform(0.3, 0.7, 2)

    u = np.zeros((ny + 2, nx + 2), dtype=np.float64)
    for p in range(n_p):
        u += amp[p] * np.exp(-((X - cx[p]) ** 2 + (Y - cy[p]) ** 2) / (2.0 * sig[p] ** 2))

    # small localized source (kept small so damping dominates -> energy decays)
    S = 1.0e-3 * np.exp(-((X - sx) ** 2 + (Y - sy) ** 2) / (2.0 * 0.04 ** 2))

    nbr = ctx["nbr"]
    if nbr["up"] == MPI.PROC_NULL:
        u[0, :] = 0.0
    if nbr["down"] == MPI.PROC_NULL:
        u[ny + 1, :] = 0.0
    if nbr["left"] == MPI.PROC_NULL:
        u[:, 0] = 0.0
    if nbr["right"] == MPI.PROC_NULL:
        u[:, nx + 1] = 0.0

    return (np.ascontiguousarray(u), np.ascontiguousarray(kappa), np.ascontiguousarray(S))


# ==============================================================================
# TASK 3: Halo pack/unpack kernels + pinned asynchronous exchange
# ==============================================================================
@cuda.jit
def pack_halos_kernel(u, s_up, s_down, s_left, s_right, ny, nx):
    k = cuda.grid(1)
    if k < nx:
        s_up[k] = u[1, k + 1]        # row 1 (contiguous)
        s_down[k] = u[ny, k + 1]     # row ny
    if k < ny:
        s_left[k] = u[k + 1, 1]      # column 1 (strided)
        s_right[k] = u[k + 1, nx]    # column nx


@cuda.jit
def unpack_halos_kernel(u, r_up, r_down, r_left, r_right, ny, nx):
    k = cuda.grid(1)
    if k < nx:
        u[0, k + 1] = r_up[k]
        u[ny + 1, k + 1] = r_down[k]
    if k < ny:
        u[k + 1, 0] = r_left[k]
        u[k + 1, nx + 1] = r_right[k]


class HaloBuffers:
    def __init__(self, ny, nx):
        n = {"up": nx, "down": nx, "left": ny, "right": ny}
        self.d_send = {d: cuda.device_array(n[d], dtype=np.float64) for d in DIRS}
        self.d_recv = {d: cuda.device_array(n[d], dtype=np.float64) for d in DIRS}
        self.h_send = {d: cuda.pinned_array(n[d], dtype=np.float64) for d in DIRS}
        self.h_recv = {d: cuda.pinned_array(n[d], dtype=np.float64) for d in DIRS}
        for d in DIRS:
            self.h_send[d][:] = 0.0
            self.h_recv[d][:] = 0.0  # stays 0 for MPI.PROC_NULL neighbours -> Dirichlet u=0
        self.threads = 128
        self.blocks = (max(ny, nx) + self.threads - 1) // self.threads


def exchange_halos_async(ctx, hb, d_u, stream):
    """Stage A (pack + async D2H on `stream`) and first half of Stage B (post Irecv/Isend).
    Returns the list of MPI requests; call finish_halo_exchange() to complete."""
    ny, nx = ctx["ny"], ctx["nx"]
    cart, nbr = ctx["cart"], ctx["nbr"]

    pack_halos_kernel[hb.blocks, hb.threads, stream](
        d_u, hb.d_send["up"], hb.d_send["down"], hb.d_send["left"], hb.d_send["right"], ny, nx
    )
    for d in DIRS:
        hb.d_send[d].copy_to_host(hb.h_send[d], stream=stream)  # async D2H into pinned memory
    stream.synchronize()  # host send buffers are now valid

    reqs = []
    for d in DIRS:  # neighbour in direction d sent us the strip it tagged TAG[OPP[d]]
        reqs.append(cart.Irecv(hb.h_recv[d], source=nbr[d], tag=TAG[OPP[d]]))
    for d in DIRS:
        reqs.append(cart.Isend(hb.h_send[d], dest=nbr[d], tag=TAG[d]))
    return reqs


def finish_halo_exchange(ctx, hb, d_u, reqs, stream):
    """Stage C (first part): Waitall, async H2D, unpack into ghost cells."""
    MPI.Request.Waitall(reqs)
    for d in DIRS:
        hb.d_recv[d].copy_to_device(hb.h_recv[d], stream=stream)
    unpack_halos_kernel[hb.blocks, hb.threads, stream](
        d_u, hb.d_recv["up"], hb.d_recv["down"], hb.d_recv["left"], hb.d_recv["right"],
        ctx["ny"], ctx["nx"],
    )


# ==============================================================================
# TASK 4: Shared-memory tiled stencil + dual-stream orchestrator
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(u_in, u_out, kappa, src, r_start, r_end, c_start, c_end,
                         dt, inv_dx2, inv_dy2, lam):
    smem = cuda.shared.array((SHARED_DIM, SHARED_DIM), dtype=float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    j = c_start + cuda.blockIdx.x * TILE_DIM + tx
    i = r_start + cuda.blockIdx.y * TILE_DIM + ty

    in_i = i <= r_end + 1
    in_j = j <= c_end + 1
    if in_i and in_j:
        smem[ty + 1, tx + 1] = u_in[i, j]              # center
        if ty == 0:
            smem[0, tx + 1] = u_in[i - 1, j]           # top apron
        if tx == 0:
            smem[ty + 1, 0] = u_in[i, j - 1]           # left apron
    if ty == TILE_DIM - 1 and (i + 1 <= r_end + 1) and in_j:
        smem[TILE_DIM + 1, tx + 1] = u_in[i + 1, j]    # bottom apron
    if tx == TILE_DIM - 1 and (j + 1 <= c_end + 1) and in_i:
        smem[ty + 1, TILE_DIM + 1] = u_in[i, j + 1]    # right apron
    cuda.syncthreads()

    if i <= r_end and j <= c_end:
        c = smem[ty + 1, tx + 1]
        n = smem[ty, tx + 1]
        s = smem[ty + 2, tx + 1]
        w = smem[ty + 1, tx]
        e = smem[ty + 1, tx + 2]
        k0 = kappa[i, j]
        k_n = 0.5 * (k0 + kappa[i - 1, j])
        k_s = 0.5 * (k0 + kappa[i + 1, j])
        k_w = 0.5 * (k0 + kappa[i, j - 1])
        k_e = 0.5 * (k0 + kappa[i, j + 1])
        rhs = ((k_s * (s - c) - k_n * (c - n)) * inv_dy2
               + (k_e * (e - c) - k_w * (c - w)) * inv_dx2
               - lam * c * c * c + src[i, j])
        u_out[i, j] = c + dt * rhs


def launch_stencil(stream, d_in, d_out, d_k, d_S, box, dt, inv_dx2, inv_dy2):
    r0, r1, c0, c1 = box
    gy = (r1 - r0 + 1 + TILE_DIM - 1) // TILE_DIM
    gx = (c1 - c0 + 1 + TILE_DIM - 1) // TILE_DIM
    tiled_stencil_kernel[(gx, gy), (TILE_DIM, TILE_DIM), stream](
        d_in, d_out, d_k, d_S, r0, r1, c0, c1, dt, inv_dx2, inv_dy2, LAMBDA_DAMP
    )


class SolverState:
    def __init__(self, ctx, u0, kappa, S, dt, dx):
        self.ctx = ctx
        ny, nx = ctx["ny"], ctx["nx"]
        self.ny, self.nx = ny, nx
        self.d_u = cuda.to_device(u0)
        self.d_un = cuda.to_device(u0.copy())
        self.d_k = cuda.to_device(kappa)
        self.d_S = cuda.to_device(S)
        self.hb = HaloBuffers(ny, nx)
        self.s_int = cuda.stream()
        self.s_bnd = cuda.stream()
        self.dt = dt
        self.inv_dx2 = 1.0 / (dx * dx)
        self.inv_dy2 = 1.0 / (dx * dx)
        self.interior_box = (2, ny - 1, 2, nx - 1)
        self.full_box = (1, ny, 1, nx)
        self.ring_boxes = [(1, 1, 1, nx), (ny, ny, 1, nx), (2, ny - 1, 1, 1), (2, ny - 1, nx, nx)]

    def swap(self):
        self.d_u, self.d_un = self.d_un, self.d_u


def step_serialized(S):
    reqs = exchange_halos_async(S.ctx, S.hb, S.d_u, S.s_bnd)
    finish_halo_exchange(S.ctx, S.hb, S.d_u, reqs, S.s_bnd)
    launch_stencil(S.s_bnd, S.d_u, S.d_un, S.d_k, S.d_S, S.full_box, S.dt, S.inv_dx2, S.inv_dy2)
    S.s_bnd.synchronize()
    S.swap()


def step_overlapped(S):
    # Stage A + B(post): pack, D2H, sync, Irecv/Isend on stream_boundary
    reqs = exchange_halos_async(S.ctx, S.hb, S.d_u, S.s_bnd)
    # Stage B: deep interior on stream_interior while MPI is in flight
    launch_stencil(S.s_int, S.d_u, S.d_un, S.d_k, S.d_S, S.interior_box, S.dt, S.inv_dx2, S.inv_dy2)
    # Stage C: wait, H2D, unpack, then 4 boundary rings on stream_boundary
    finish_halo_exchange(S.ctx, S.hb, S.d_u, reqs, S.s_bnd)
    for box in S.ring_boxes:
        launch_stencil(S.s_bnd, S.d_u, S.d_un, S.d_k, S.d_S, box, S.dt, S.inv_dx2, S.inv_dy2)
    S.s_int.synchronize()
    S.s_bnd.synchronize()
    S.swap()


# ==============================================================================
# TASK 5: reductions, benchmark, PTX audit, manifest
# ==============================================================================
def run_mode(ctx, mode, steps, u0, kappa, S0, dt, dx, verbose=True):
    cart = ctx["cart"]
    ny, nx = ctx["ny"], ctx["nx"]
    st = SolverState(ctx, u0, kappa, S0, dt, dx)
    step_fn = step_serialized if mode == "serialized" else step_overlapped
    dxdy = dx * dx

    def local_energy(u_host):
        return float(u_host[1:ny + 1, 1:nx + 1].sum() * dxdy)

    trace = []
    e0 = allreduce_scalar(cart, local_energy(st.d_u.copy_to_host()), MPI.SUM)
    trace.append({"step": 0, "energy": e0, "l2_residual": 0.0})

    wall_s = 0.0
    event_ms = 0.0
    ev_a, ev_b = cuda.event(), cuda.event()
    cart.Barrier()
    for s in range(1, steps + 1):
        report = (s % REPORT_EVERY == 0)
        old = st.d_u.copy_to_host() if report else None
        t0 = time.perf_counter()
        ev_a.record(stream=st.s_bnd)
        step_fn(st)
        ev_b.record(stream=st.s_bnd)
        ev_b.synchronize()
        wall_s += time.perf_counter() - t0
        event_ms += cuda.event_elapsed_time(ev_a, ev_b)
        if report:
            new = st.d_u.copy_to_host()
            e = allreduce_scalar(cart, local_energy(new), MPI.SUM)
            diff = new[1:ny + 1, 1:nx + 1] - old[1:ny + 1, 1:nx + 1]
            l2 = math.sqrt(allreduce_scalar(cart, float((diff * diff).sum()), MPI.SUM))
            trace.append({"step": s, "energy": e, "l2_residual": l2})
    cart.Barrier()

    wall_ms = allreduce_scalar(cart, wall_s * 1e3, MPI.MAX)
    ev_ms = allreduce_scalar(cart, event_ms, MPI.MAX)
    return {"u": st.d_u.copy_to_host(), "trace": trace, "wall_ms": wall_ms, "event_ms": ev_ms}


def ptx_audit():
    asm = tiled_stencil_kernel.inspect_asm()
    if isinstance(asm, dict):
        asm = "\n".join(asm.values())
    counts = {
        "st.shared.f64": len(re.findall(r"\bst\.shared\.f64\b", asm)),
        "ld.shared.f64": len(re.findall(r"\bld\.shared\.f64\b", asm)),
        "bar.sync": len(re.findall(r"\bbar\.sync\b", asm)),
    }
    return counts, all(v > 0 for v in counts.values())


def run_solver(github_user: str):
    ctx = init_distributed_context(github_user)
    cart, rank = ctx["cart"], ctx["rank"]
    u0, kappa, S0 = initialize_subdomains(ctx)

    dx = 1.0 / (GLOBAL_NX + 1)
    kappa_max = allreduce_scalar(cart, float(kappa.max()), MPI.MAX)
    dt_max = dx * dx / (4.0 * kappa_max)
    dt = DT_SAFETY * dt_max

    if rank == 0:
        print(f"GITHUB_USERNAME     : {ctx['github_user']}")
        print(f"GPU UUID            : {ctx['gpu_uuid']}")
        print(f"INSTANCE_SEED       : {ctx['seed']}")
        print(f"MPI ranks / grid    : {ctx['size']} / 2x2 Cartesian (non-periodic)")
        print(f"kappa_max={kappa_max:.6f}  dx={dx:.8e}  dt_max={dt_max:.6e}  dt={dt:.6e}")
    for r in range(4):
        cart.Barrier()
        if r == rank:
            print(f"[rank {rank}] coords=({ctx['row_rank']},{ctx['col_rank']}) nbrs={ctx['nbr']}")
    cart.Barrier()

    # JIT warm-up (excluded from timings)
    run_mode(ctx, "serialized", 2, u0, kappa, S0, dt, dx)
    run_mode(ctx, "overlapped", 2, u0, kappa, S0, dt, dx)

    ser = run_mode(ctx, "serialized", STEPS, u0, kappa, S0, dt, dx)
    ovl = run_mode(ctx, "overlapped", STEPS, u0, kappa, S0, dt, dx)

    ny, nx = ctx["ny"], ctx["nx"]
    local_diff = float(np.abs(ser["u"][1:ny + 1, 1:nx + 1] - ovl["u"][1:ny + 1, 1:nx + 1]).max())
    parity = allreduce_scalar(cart, local_diff, MPI.MAX)

    counts, ptx_ok = ptx_audit()

    energies = [t["energy"] for t in ovl["trace"]]
    monotone = all(energies[k + 1] <= energies[k] for k in range(len(energies) - 1))

    if rank == 0:
        print("\n--- Allreduce energy / L2 residual trace (overlapped) ---")
        for t in ovl["trace"]:
            print(f"step {t['step']:4d}  E = {t['energy']:.10e}  ||du||_2 = {t['l2_residual']:.6e}")
        print(f"monotone energy decay: {monotone}")
        print("\n--- Serialized trace ---")
        for t in ser["trace"]:
            print(f"step {t['step']:4d}  E = {t['energy']:.10e}  ||du||_2 = {t['l2_residual']:.6e}")
        print(f"\nserialized_ms = {ser['wall_ms']:.3f} (CUDA events: {ser['event_ms']:.3f})")
        print(f"overlapped_ms = {ovl['wall_ms']:.3f} (CUDA events: {ovl['event_ms']:.3f})")
        print(f"speedup (serialized/overlapped) = {ser['wall_ms'] / ovl['wall_ms']:.3f}x")
        print(f"parity_max_diff = {parity:.3e}  (required < 1e-12)")
        print(f"PTX counts: {counts}  ptx_audit_passed={ptx_ok}")

        manifest = {
            "GITHUB_USERNAME": ctx["github_user"],
            "gpu_uuid": ctx["gpu_uuid"],
            "instance_seed": ctx["seed"],
            "mpi_ranks": ctx["size"],
            "cartesian_dims": [2, 2],
            "global_grid": [GLOBAL_NY, GLOBAL_NX],
            "local_padded_shape": [ny + 2, nx + 2],
            "steps": STEPS,
            "kappa_max": kappa_max,
            "dx": dx,
            "dt_max": dt_max,
            "dt": dt,
            "serialized_ms": ser["wall_ms"],
            "overlapped_ms": ovl["wall_ms"],
            "serialized_cuda_event_ms": ser["event_ms"],
            "overlapped_cuda_event_ms": ovl["event_ms"],
            "parity_max_diff": parity,
            "ptx_counts": counts,
            "ptx_audit_passed": bool(ptx_ok),
            "energy_trace": ovl["trace"],
            "energy_monotone_decreasing": bool(monotone),
        }
        with open("submission_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print("\nWrote submission_manifest.json")

    if parity >= 1e-12 or not ptx_ok:
        raise RuntimeError("Verification failed (parity or PTX audit).")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: mpirun -np 4 python3 distributed_cuda_solver.py <github_username>")
    run_solver(sys.argv[1])
