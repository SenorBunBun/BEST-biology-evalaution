#!/usr/bin/env python3
"""
Batch Runner for All Dataset Configs

Runs the full pipeline (train -> generate -> analyze) for all OSD dataset configs.

Usage:
    python run_all_datasets.py                    # Run all OSD configs
    python run_all_datasets.py --configs osdr-402 osdr-403  # Run specific configs
    python run_all_datasets.py --stage train      # Run only training stage
    python run_all_datasets.py --show-plots       # Show plots during analysis
    python run_all_datasets.py --dry-run          # Preview what would run
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Get script directory
SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_DIR = SCRIPT_DIR / "dataset_configs"
DRIVER_PATH = SCRIPT_DIR / "driver.py"

# Default OSD configs to run (in order)
DEFAULT_CONFIGS = [
    "osdr-402",
    "osdr-403",
    "osdr-404",
    "osdr-405",
]

# All available configs
ALL_CONFIGS = [
    "osdr-402",
    "osdr-403",
    "osdr-404",
    "osdr-405",
    "osdr-40X",  # Combined dataset
    "spaceflight",
    "cancer",
]


def clean_output_folders(config_name: str, dry_run: bool = False) -> bool:
    """Clean synthetic_data and analysis folders for a config."""
    config_path = get_config_path(config_name)

    # Load config to get output directory
    with open(config_path, 'r') as f:
        config = json.load(f)

    output_dir = Path(config.get('paths', {}).get('output_dir', f'outputs/{config_name}'))
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir

    folders_to_clean = [
        output_dir / 'synthetic_data',
        output_dir / 'analysis',
    ]

    cleaned = False
    for folder in folders_to_clean:
        if folder.exists():
            if dry_run:
                print(f"  [DRY RUN] Would delete: {folder}")
            else:
                import shutil
                try:
                    shutil.rmtree(folder)
                    folder.mkdir(parents=True, exist_ok=True)
                    print(f"  [CLEANED] {folder}")
                    cleaned = True
                except Exception as e:
                    print(f"  [ERROR] Failed to clean {folder}: {e}")

    if not cleaned and not dry_run:
        print(f"  [INFO] No folders to clean for {config_name}")

    return True


def get_config_path(config_name: str) -> Path:
    """Get full path to config file."""
    if config_name.endswith('.json'):
        config_path = Path(config_name)
    else:
        config_path = CONFIG_DIR / f"{config_name}.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    return config_path


def run_pipeline(config_name: str, stage: str = None, model: str = None,
                 show_plots: bool = False, fail_fast: bool = True,
                 dry_run: bool = False, extra_args: list = None) -> bool:
    """
    Run the pipeline for a single config.

    Returns:
        True if successful, False otherwise
    """
    config_path = get_config_path(config_name)

    # Build command
    cmd = [
        sys.executable,
        str(DRIVER_PATH),
        "--config", str(config_path),
    ]

    if stage:
        cmd.extend(["--stage", stage])

    if model:
        cmd.extend(["--model", model])

    if fail_fast:
        cmd.append("--fail-fast")

    if show_plots:
        cmd.append("--show-plots")

    if extra_args:
        cmd.extend(extra_args)

    # Print command
    print("\n" + "=" * 70)
    print(f"RUNNING: {config_name}")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print("-" * 70)

    if dry_run:
        print("[DRY RUN] Would execute above command")
        return True

    # Run command
    start_time = datetime.now()

    try:
        # Set environment for UTF-8 output
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        result = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            env=env,
            # Don't capture output - let it stream to console
        )

        elapsed = datetime.now() - start_time

        if result.returncode == 0:
            print(f"\n[SUCCESS] {config_name} completed in {elapsed}")
            return True
        else:
            print(f"\n[FAILED] {config_name} failed with exit code {result.returncode}")
            return False

    except Exception as e:
        print(f"\n[ERROR] {config_name} failed with exception: {e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch run pipeline for multiple dataset configs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_datasets.py                         # Run all default OSD configs
  python run_all_datasets.py --all                   # Run ALL configs (including cancer, spaceflight)
  python run_all_datasets.py --configs osdr-402 osdr-403  # Run specific configs
  python run_all_datasets.py --stage train           # Run only training
  python run_all_datasets.py --stage generate        # Run only generation
  python run_all_datasets.py --stage analyze         # Run only analysis
  python run_all_datasets.py --regenerate            # Run generate + analyze (skip training)
  python run_all_datasets.py --regenerate --clean    # Clean old files, then regenerate
  python run_all_datasets.py --regenerate --batch-stages --clean  # Full batch regenerate
  python run_all_datasets.py --show-plots            # Show plots during analysis
  python run_all_datasets.py --dry-run               # Preview commands
  python run_all_datasets.py --continue-on-error     # Don't stop on failures
        """
    )

    parser.add_argument('--configs', nargs='+', default=None,
                        help='Specific configs to run (default: all OSD configs)')
    parser.add_argument('--all', action='store_true',
                        help='Run ALL configs including cancer and spaceflight')
    parser.add_argument('--stage', choices=['train', 'generate', 'analyze'],
                        help='Run only a specific stage')
    parser.add_argument('--regenerate', action='store_true',
                        help='Run generate + analyze stages only (skip training)')
    parser.add_argument('--batch-stages', action='store_true',
                        help='With --regenerate: run generate for ALL configs first, then analyze for ALL')
    parser.add_argument('--model', choices=['cvae', 'diffusion', 'gan'],
                        help='Run only a specific model (requires --stage)')
    parser.add_argument('--show-plots', action='store_true',
                        help='Show plots during analysis')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview commands without running')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Continue running other configs if one fails')
    parser.add_argument('--list', action='store_true',
                        help='List available configs and exit')
    parser.add_argument('--clean', action='store_true',
                        help='Clear synthetic_data/ and analysis/ folders before running')

    return parser.parse_args()


def main():
    args = parse_args()

    # List configs and exit
    if args.list:
        print("Available configs:")
        print("-" * 40)
        for config in ALL_CONFIGS:
            config_path = CONFIG_DIR / f"{config}.json"
            status = "[OK]" if config_path.exists() else "[MISSING]"
            default = "(default)" if config in DEFAULT_CONFIGS else ""
            print(f"  {status} {config} {default}")
        return 0

    # Determine which configs to run
    if args.configs:
        configs_to_run = args.configs
    elif args.all:
        configs_to_run = ALL_CONFIGS
    else:
        configs_to_run = DEFAULT_CONFIGS

    # Validate configs exist
    valid_configs = []
    for config in configs_to_run:
        try:
            get_config_path(config)
            valid_configs.append(config)
        except FileNotFoundError as e:
            print(f"[WARN] Skipping {config}: {e}")

    if not valid_configs:
        print("[ERROR] No valid configs to run")
        return 1

    # Determine stages to run
    if args.regenerate:
        stages_to_run = ['generate', 'analyze']
        stage_mode = "regenerate (generate + analyze)"
    elif args.stage:
        stages_to_run = [args.stage]
        stage_mode = args.stage
    else:
        stages_to_run = None  # Full pipeline
        stage_mode = "all"

    # Print summary
    print("=" * 70)
    print("BATCH PIPELINE RUNNER")
    print("=" * 70)
    print(f"Configs to run: {', '.join(valid_configs)}")
    print(f"Stage: {stage_mode}")
    print(f"Model: {args.model or 'all'}")
    if args.regenerate and args.batch_stages:
        print(f"Batch mode: generate ALL first, then analyze ALL")
    print(f"Show plots: {args.show_plots}")
    print(f"Continue on error: {args.continue_on_error}")
    print(f"Clean folders: {args.clean}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 70)

    # Clean folders if requested
    if args.clean:
        print("\n" + "-" * 70)
        print("CLEANING OUTPUT FOLDERS")
        print("-" * 70)
        for config in valid_configs:
            print(f"\nCleaning {config}...")
            clean_output_folders(config, dry_run=args.dry_run)

    # Run each config
    results = {}
    start_time = datetime.now()

    # Batch stages mode: run each stage for all configs before moving to next stage
    if args.regenerate and args.batch_stages:
        for stage in stages_to_run:
            print(f"\n{'='*70}")
            print(f"BATCH STAGE: {stage.upper()}")
            print(f"{'='*70}")

            for i, config in enumerate(valid_configs, 1):
                print(f"\n[{i}/{len(valid_configs)}] {stage.upper()}: {config}...")

                success = run_pipeline(
                    config_name=config,
                    stage=stage,
                    model=args.model,
                    show_plots=args.show_plots if stage == 'analyze' else False,
                    fail_fast=not args.continue_on_error,
                    dry_run=args.dry_run,
                )

                key = f"{config}_{stage}"
                results[key] = success

                if not success and not args.continue_on_error:
                    print(f"\n[STOPPED] Stopping due to failure (use --continue-on-error to continue)")
                    break

            # Check if we should continue to next stage
            stage_failed = any(not v for k, v in results.items() if stage in k)
            if stage_failed and not args.continue_on_error:
                break

    # Sequential mode: run all stages for each config before moving to next config
    else:
        for i, config in enumerate(valid_configs, 1):
            print(f"\n[{i}/{len(valid_configs)}] Processing {config}...")

            if stages_to_run:
                # Run specific stages
                config_success = True
                for stage in stages_to_run:
                    success = run_pipeline(
                        config_name=config,
                        stage=stage,
                        model=args.model,
                        show_plots=args.show_plots if stage == 'analyze' else False,
                        fail_fast=not args.continue_on_error,
                        dry_run=args.dry_run,
                    )
                    if not success:
                        config_success = False
                        if not args.continue_on_error:
                            break
                results[config] = config_success
            else:
                # Run full pipeline
                success = run_pipeline(
                    config_name=config,
                    stage=None,
                    model=args.model,
                    show_plots=args.show_plots,
                    fail_fast=not args.continue_on_error,
                    dry_run=args.dry_run,
                )
                results[config] = success

            if not results[config] and not args.continue_on_error:
                print(f"\n[STOPPED] Stopping due to failure (use --continue-on-error to continue)")
                break

    # Print summary
    total_time = datetime.now() - start_time

    print("\n" + "=" * 70)
    print("BATCH RUN SUMMARY")
    print("=" * 70)

    successful = [c for c, s in results.items() if s]
    failed = [c for c, s in results.items() if not s]

    print(f"Total time: {total_time}")
    print(f"Successful: {len(successful)}/{len(results)}")

    if successful:
        print(f"  [OK] {', '.join(successful)}")
    if failed:
        print(f"  [FAILED] {', '.join(failed)}")

    print("=" * 70)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
