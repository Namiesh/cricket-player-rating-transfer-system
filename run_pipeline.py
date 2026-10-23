"""
Pipeline Execution Script.
Executes the full cricket player rating & transfer value system pipeline,
trains CatBoost regressors, computes market valuations, and exports reports.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding='utf-8')

from src.pipeline import CricketPipelineOrchestrator


def main():
    parser = argparse.ArgumentParser(description="Run Cricket Rating and Valuation Pipeline")
    parser.add_argument("--matches", type=int, default=1500, help="Number of matches to ingest (default: 1500)")
    args = parser.parse_args()

    orchestrator = CricketPipelineOrchestrator()
    orchestrator.run(max_matches=args.matches)


if __name__ == "__main__":
    main()
