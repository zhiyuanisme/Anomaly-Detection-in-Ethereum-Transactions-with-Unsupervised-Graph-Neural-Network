#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化异常检测流水线 / Automated Anomaly Detection Pipeline
Ethereum Transaction Anomaly Detection with Unsupervised Graph Neural Networks

Usage:
    python pipeline.py --stage all          # Run the full pipeline
    python pipeline.py --stage check        # Check environment and data
    python pipeline.py --stage preprocess   # Preprocess data
    python pipeline.py --stage evaluate     # Run evaluation and generate reports
"""

import argparse
import os
import sys
import json
import logging
import time
import pickle
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    """Print a decorated section header."""
    border = "=" * 60
    logger.info(border)
    logger.info(f"  {title}")
    logger.info(border)


def _check_file(path: str) -> bool:
    """Return True and log success if *path* exists, else log a warning."""
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        logger.info(f"  ✅ {path}  ({size_kb:.1f} KB)")
        return True
    logger.warning(f"  ⚠️  Missing: {path}")
    return False


# ---------------------------------------------------------------------------
# Stage 1 – Environment & data check
# ---------------------------------------------------------------------------

def stage_check() -> bool:
    """Verify that the required Python packages and data files are available."""
    _section("Stage 1: Environment & Data Check")
    all_ok = True

    # --- Python packages ---
    logger.info("Checking required Python packages …")
    required_packages = [
        ("torch", "PyTorch"),
        ("torch_geometric", "PyTorch Geometric"),
        ("sklearn", "scikit-learn"),
        ("networkx", "NetworkX"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
    ]
    for module_name, display_name in required_packages:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "unknown")
            logger.info(f"  ✅ {display_name} ({version})")
        except ImportError:
            logger.warning(f"  ⚠️  {display_name} not installed (pip install {module_name})")
            all_ok = False

    # --- Data files ---
    logger.info("Checking expected data files …")
    data_paths = [
        "Dataset/MulDiGraph.pkl",
        "Dataset/test_subgraph.pkl",
        "Dataset/Processed/phishing_nodes.pkl",
    ]
    for p in data_paths:
        if not _check_file(p):
            all_ok = False

    # --- Result artefacts (non-fatal) ---
    logger.info("Checking pre-computed result artefacts (optional) …")
    result_paths = [
        "top_100_global_indices.npy",
        "top_500_global_indices.npy",
        "top_1000_global_indices.npy",
        "top_2000_global_indices.npy",
        "top_5000_global_indices.npy",
        "top_10000_global_indices.npy",
        "full_global_ranking.npy",
        "global_performance_summary.json",
        "comprehensive_global_performance_results.csv",
    ]
    for p in result_paths:
        _check_file(p)

    if all_ok:
        logger.info("✅ Environment check passed.")
    else:
        logger.warning("⚠️  Some required items are missing – subsequent stages may fail.")
    return all_ok


# ---------------------------------------------------------------------------
# Stage 2 – Data pre-processing
# ---------------------------------------------------------------------------

def stage_preprocess() -> bool:
    """Run the compatibility-file generation helper and basic data validation."""
    _section("Stage 2: Data Pre-processing")

    # Run the training-data checker that is already part of the repo
    logger.info("Running check_training_data …")
    try:
        from check_training_data import check_training_data
        check_training_data()
        logger.info("  Training-data check completed.")
    except Exception as exc:
        logger.warning(f"  check_training_data skipped: {exc}")

    # Attempt to build SuperGAT compatibility files if the source data exists
    supergat_topk_dir = os.path.join("supergat_autoencoder_ocsvm", "topk_results")
    if os.path.isdir(supergat_topk_dir):
        logger.info("SuperGAT topk results found – generating compatibility files …")
        try:
            from create_supergat_compat_files import create_supergat_compat_files
            success = create_supergat_compat_files()
            if success:
                logger.info("  ✅ SuperGAT compatibility files created.")
            else:
                logger.warning("  ⚠️  SuperGAT compatibility file creation failed.")
        except Exception as exc:
            logger.warning(f"  create_supergat_compat_files skipped: {exc}")
    else:
        logger.info("  SuperGAT topk results directory not found – skipping compat file creation.")

    logger.info("✅ Pre-processing stage completed.")
    return True


# ---------------------------------------------------------------------------
# Stage 3 – Evaluation & report generation
# ---------------------------------------------------------------------------

def stage_evaluate() -> bool:
    """Load pre-computed results and produce a consolidated JSON summary."""
    _section("Stage 3: Evaluation & Report Generation")

    report: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "models": {},
    }

    # ---- GraphSAGE results ------------------------------------------------
    graphsage_dir = os.path.join("graphsage_autoencoder_ocsvm", "topk_results")
    graphsage_quick = os.path.join(graphsage_dir, "quick_ocsvm_results.pkl")
    if os.path.exists(graphsage_quick):
        try:
            with open(graphsage_quick, "rb") as fh:
                gs_data = pickle.load(fh)
            report["models"]["graphsage"] = {
                "precision_100": gs_data.get("precision_100"),
                "top_100_anomaly_count": gs_data.get("top_100_anomaly_count"),
                "model_type": gs_data.get("model_type", "GraphSAGE_Autoencoder_OCSVM"),
            }
            logger.info(f"  GraphSAGE  Top-100 precision: {gs_data.get('precision_100', 'N/A')}")
        except Exception as exc:
            logger.warning(f"  GraphSAGE results load failed: {exc}")
    else:
        logger.info("  GraphSAGE quick results not found – skipping.")

    # ---- SuperGAT results -------------------------------------------------
    supergat_dir = os.path.join("supergat_autoencoder_ocsvm", "topk_results")
    supergat_quick = os.path.join(supergat_dir, "quick_ocsvm_results.pkl")
    if os.path.exists(supergat_quick):
        try:
            with open(supergat_quick, "rb") as fh:
                sg_data = pickle.load(fh)
            report["models"]["supergat"] = {
                "precision_100": sg_data.get("precision_100"),
                "top_100_anomaly_count": sg_data.get("top_100_anomaly_count"),
                "model_type": sg_data.get("model_type", "SuperGAT_Autoencoder_OCSVM"),
            }
            logger.info(f"  SuperGAT   Top-100 precision: {sg_data.get('precision_100', 'N/A')}")
        except Exception as exc:
            logger.warning(f"  SuperGAT results load failed: {exc}")
    else:
        logger.info("  SuperGAT quick results not found – skipping.")

    # ---- Global performance summary (pre-computed) ------------------------
    if os.path.exists("global_performance_summary.json"):
        try:
            with open("global_performance_summary.json") as fh:
                global_summary = json.load(fh)
            report["global_performance"] = global_summary
            logger.info("  Global performance summary loaded.")
        except Exception as exc:
            logger.warning(f"  global_performance_summary load failed: {exc}")

    # ---- Save report ------------------------------------------------------
    report_path = f"pipeline_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logger.info(f"✅ Evaluation report saved → {report_path}")
    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

STAGES = {
    "check": stage_check,
    "preprocess": stage_preprocess,
    "evaluate": stage_evaluate,
}


def run_all() -> bool:
    """Execute all pipeline stages sequentially."""
    results = {}
    for name, fn in STAGES.items():
        start = time.time()
        ok = fn()
        elapsed = time.time() - start
        results[name] = {"ok": ok, "elapsed_s": round(elapsed, 2)}
        if not ok:
            logger.error(f"Stage '{name}' reported errors.  Continuing …")

    _section("Pipeline Summary")
    all_ok = True
    for name, info in results.items():
        icon = "✅" if info["ok"] else "❌"
        logger.info(f"  {icon}  {name:<15} ({info['elapsed_s']} s)")
        if not info["ok"]:
            all_ok = False

    if all_ok:
        logger.info("🎉 Pipeline completed successfully.")
    else:
        logger.warning("⚠️  Pipeline completed with warnings/errors.")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated pipeline for Ethereum anomaly detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        choices=list(STAGES) + ["all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    args = parser.parse_args()

    if args.stage == "all":
        ok = run_all()
    else:
        ok = STAGES[args.stage]()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
