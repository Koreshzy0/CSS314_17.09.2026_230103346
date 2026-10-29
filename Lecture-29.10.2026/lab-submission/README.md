# CSS314 Parallel Computing and Performance Optimization

## Summary

This practicum measures memory stride performance, loop interchange, cache-line false sharing, strong and weak scaling, and stencil arithmetic intensity. The experiments record host hardware information and student-specific parameters. Results are generated on the local machine and may vary depending on processor topology, cache hierarchy, memory bandwidth, operating system, and process scheduling.

## Requirements

- Python 3
- NumPy
- psutil
- Numba

## Run

```bash
python3 -m pip install numpy psutil numba
python3 src/profile_env.py --student-id 30103346
python3 src/stride_bench.py
python3 src/dense_kernel.py
python3 src/false_sharing.py
python3 src/scaling_study.py
python3 verify_outputs.py
```

## Results

All generated measurements are stored in `results/`.

The experiments should be interpreted using the actual recorded timings. A speedup or bottleneck classification must not be assumed before the measurements are collected.