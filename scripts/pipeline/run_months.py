#!/usr/bin/env python3
"""
CLI Runner for OceanEmbed Month Pipeline

Iterates over month range [start_month..end_month] and executes the pipeline driver.
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline.month_pipeline import (
    DEFAULT_INPUT_POLICY,
    DEFAULT_QC_POLICY,
    DEFAULT_SOURCES_CONFIG,
    DEFAULT_TARGET_POLICY,
    PROJECT_ROOT,
    generate_month_range,
    load_sources_config,
    process_month,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OceanEmbed Monthly Preprocessing Pipeline Driver"
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start month in YYYY-MM format (e.g. 2019-01)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End month in YYYY-MM format (inclusive). Defaults to --start.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        default=False,
        help="Attempt download of missing raw files (mocked in offline mode).",
    )
    parser.add_argument(
        "--use-existing-raw",
        action="store_true",
        default=False,
        help="Use existing raw files on disk without downloading.",
    )
    parser.add_argument(
        "--delete-raw-glorys",
        action="store_true",
        default=False,
        help="Delete raw GLORYS reanalysis file ONLY after successful build and manifest write.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-execution even if month already marked OK in manifest.",
    )
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        default=False,
        help="Continue processing subsequent months if a month fails.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(PROJECT_ROOT, "data", "processed"),
        help="Output root directory for processed files and manifest (default: data/processed).",
    )
    parser.add_argument(
        "--config-sources",
        default=DEFAULT_SOURCES_CONFIG,
        help=f"Path to sources config JSON (default: {DEFAULT_SOURCES_CONFIG}).",
    )
    parser.add_argument(
        "--qc-policy",
        default=DEFAULT_QC_POLICY,
        help=f"Path to GLORYS QC policy JSON (default: {DEFAULT_QC_POLICY}).",
    )
    parser.add_argument(
        "--target-policy",
        default=DEFAULT_TARGET_POLICY,
        help=f"Path to GLORYS target policy JSON (default: {DEFAULT_TARGET_POLICY}).",
    )
    parser.add_argument(
        "--input-policy",
        default=DEFAULT_INPUT_POLICY,
        help=f"Path to input policy JSON (default: {DEFAULT_INPUT_POLICY}).",
    )
    parser.add_argument(
        "--log-dir",
        default=os.path.join(PROJECT_ROOT, "outputs", "pipeline", "logs"),
        help="Directory to save execution logs.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    start_month = args.start
    end_month = args.end if args.end is not None else start_month

    logger, log_path = setup_logger(args.log_dir)
    logger.info("================================================================================")
    logger.info("OceanEmbed Monthly Preprocessing Pipeline Driver v1")
    logger.info("================================================================================")
    logger.info(f"Month Range: {start_month} to {end_month}")
    logger.info(f"Output Root: {args.output_root}")
    logger.info(f"Options: download={args.download}, use_existing_raw={args.use_existing_raw}, "
                f"delete_raw_glorys={args.delete_raw_glorys}, force={args.force}, "
                f"continue_on_fail={args.continue_on_fail}")
    logger.info(f"Log File: {log_path}")

    # Load configuration
    sources_cfg = load_sources_config(args.config_sources)
    months = generate_month_range(start_month, end_month)
    logger.info(f"Target Months ({len(months)}): {', '.join(months)}")

    manifest_csv = os.path.join(args.output_root, "months_manifest.csv")
    overall_success = True
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for m in months:
        logger.info("-" * 80)
        try:
            res = process_month(
                month=m,
                sources_cfg=sources_cfg,
                config_path=args.config_sources,
                qc_policy_path=args.qc_policy,
                target_policy_path=args.target_policy,
                input_policy_path=args.input_policy,
                output_root=args.output_root,
                download=args.download,
                use_existing_raw=args.use_existing_raw,
                delete_raw_glorys=args.delete_raw_glorys,
                force=args.force,
                continue_on_fail=args.continue_on_fail,
                manifest_csv_path=manifest_csv,
                logger=logger,
            )

            status = res.get("status")
            if status == "OK":
                processed_count += 1
            elif status == "SKIPPED":
                skipped_count += 1
            else:
                failed_count += 1
                overall_success = False

        except Exception as e:
            logger.error(f"Execution halted on month {m}: {e}")
            failed_count += 1
            overall_success = False
            if not args.continue_on_fail:
                logger.error("Stopping run due to error (use --continue-on-fail to proceed past errors).")
                return 1

    logger.info("=" * 80)
    logger.info("Pipeline Run Summary")
    logger.info("=" * 80)
    logger.info(f"Total Months: {len(months)}")
    logger.info(f"Processed OK: {processed_count}")
    logger.info(f"Skipped     : {skipped_count}")
    logger.info(f"Failed      : {failed_count}")
    logger.info(f"Manifest    : {manifest_csv}")
    logger.info(f"Log Path    : {log_path}")

    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
