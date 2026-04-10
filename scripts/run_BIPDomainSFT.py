#!/usr/bin/env python
"""
CLI entry point for BIPDomainSFT training.
Submitted via sbatch scripts/train.slurm.

Usage:
    python scripts/run_BIPDomainSFT.py --data data/raw/bips.csv
    python scripts/run_BIPDomainSFT.py --data data/raw/bips.csv --smoke_test
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).resolve().parents[1] / "src" / "training" / "BIPDomainSFT.py"
    cmd = [sys.executable, str(script)] + sys.argv[1:]
    subprocess.run(cmd, check=True)