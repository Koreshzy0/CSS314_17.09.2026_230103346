
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np


def main():
    params = json.loads(
        Path("results/salted_params.json").read_text()
    )
    count = params["buffer_elements"]
    data = np.ones(count, dtype=np.float64)
    strides = [1, 2, 4, 8, 16, 32, 64, 128]
    output = []

    for stride in strides:
        def traverse():
            return np.sum(data[::stride], dtype=np.float64)

        for _ in range(5):
            traverse()

        timings = []
        for _ in range(5):
            start = time.perf_counter_ns()
            total = traverse()
            timings.append(time.perf_counter_ns() - start)

        median_ns = statistics.median(timings)
        touched = len(data[::stride])

        output.append({
            "stride": stride,
            "elements_touched": touched,
            "median_time_ns": median_ns,
            "ns_per_element": median_ns / touched,
            "elements_per_second": touched / (median_ns / 1e9),
            "checksum": float(total),
        })

    Path("results").mkdir(exist_ok=True)
    with open("results/stride_results.tsv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=output[0].keys(), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(output)

    print(*output, sep="\n")


if __name__ == "__main__":
    main()
