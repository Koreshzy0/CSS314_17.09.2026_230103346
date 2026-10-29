import json
from pathlib import Path

def main():
    results = Path("results")

    required = [
        "host_profile.json",
        "salted_params.json",
        "stride_results.tsv",
        "loop_interchange.json",
        "false_sharing_karp_flatt.tsv",
        "scaling_metrics.tsv",
        "roofline_summary.json",
    ]

    missing = [
        name for name in required
        if not (results / name).exists()
    ]

    if missing:
        print("[FAIL] Missing result files:")
        for name in missing:
            print(" -", name)
        print("\nRun the tasks and check that each script saves its output.")
        return

    profile = json.loads(
        (results / "host_profile.json").read_text()
    )
    params = json.loads(
        (results / "salted_params.json").read_text()
    )

    assert len(params["student_id"]) == 8
    assert 1024 <= params["N"] <= 1536
    assert params["N"] % 64 == 0
    assert params["buffer_elements"] == 16_777_216

    print("[PASS] Required result files exist.")
    print("[PASS] Student parameters are valid.")
    print("CPU:", profile["cpu"]["model"])
    print("Student ID:", params["student_id"])

if __name__ == "__main__":
    main()
