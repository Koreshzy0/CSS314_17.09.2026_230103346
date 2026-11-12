# Parallel Computing Laboratory

**Student ID:** 230103346

## Project Overview

This project contains experiments on parallel computing,
performance measurement, CPU architecture and scheduling.

## Project Structure

- task0_cal.py — Hardware baseline
- task1_falsesharing.py — Phase 1: False Sharing
- task2_scaling.py — Phase 2: Strong and Weak Scaling
- task3a_roofline.py — Phase 3: Empirical Roofline
- task3b_gemm.py — Phase 3: GEMM Optimization
- task4_scheduling.py — Phase 4: Dynamic Scheduling
- harness.py — Benchmarking and verification utilities

## Experimental Results

- phase1_results.json
- phase2_results.json
- phase3a_results.json
- phase3b_results.json
- phase4_results.json

## Data Tables

- table1a.csv
- table1b.csv

## Requirements

Python 3

## Run Phase 4

```bash
python3 task4_scheduling.py

```bash
cat > .gitignore <<'EOF'
__pycache__/
*.py[cod]
.DS_Store
.venv/
venv/
