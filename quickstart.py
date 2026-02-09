#!/usr/bin/env python3
"""
Quick start for translation/resume only.
"""

import sys
from pathlib import Path
import logging

# Add code directory to path
CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))

from main import SafetyEvaluationPipeline, EvaluationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def translate_dataset(dataset_limit: int | None = None):
    """
    Option 1: Translate dataset.json only (step 1).
    """
    logger.info("\n" + "="*80)
    logger.info("TRANSLATE DATASET (STEP 1 ONLY)")
    logger.info("="*80)

    config = EvaluationConfig()
    pipeline = SafetyEvaluationPipeline(
        config=config,
        output_dir="./results_translation"
    )

    dataset = pipeline.step_1_create_dataset(dataset_limit=dataset_limit)
    logger.info(f"✓ Dataset created with {len(dataset.prompts)} prompts")
    logger.info(f"Results saved to: {pipeline.run_dir}")


def resume_from_translation(dataset_limit: int | None = None):
    """
    Option 2: Resume from an existing translated dataset.json (step 2+).
    """
    logger.info("\n" + "="*80)
    logger.info("RESUME FROM TRANSLATION (STEP 2+)")
    logger.info("="*80)

    dataset_path = input("Enter path to translated dataset.json: ").strip()
    if not dataset_path:
        logger.error("No dataset path provided.")
        return

    config = EvaluationConfig()
    pipeline = SafetyEvaluationPipeline(
        config=config,
        output_dir="./results_resume"
    )

    pipeline.run_from_step(start_step=2, dataset_path=dataset_path, dataset_limit=dataset_limit)
    logger.info(f"\nResults saved to: {pipeline.run_dir}")


def print_menu():
    """Print menu."""
    print("\n" + "="*80)
    print("MULTILINGUAL AI SAFETY EVALUATION - QUICK START")
    print("="*80 + "\n")
    print("Select an option:\n")
    print("1. Translate dataset.json (step 1 only)")
    print("2. Resume from translated dataset.json (step 2+)")
    print("3a. Translate first 5 items (verbose logs)")
    print("3b. Resume first 5 items (verbose logs)")
    print()


if __name__ == "__main__":
    print_menu()

    try:
        choice = input("Enter your choice (1-2, 3a, 3b): ").strip().lower()

        if choice == "1":
            translate_dataset()
        elif choice == "2":
            resume_from_translation()
        elif choice == "3a":
            logging.getLogger().setLevel(logging.DEBUG)
            translate_dataset(dataset_limit=1)
        elif choice == "3b":
            logging.getLogger().setLevel(logging.DEBUG)
            resume_from_translation(dataset_limit=1)
        else:
            print("Invalid choice. Please enter 1-2, 3a, or 3b.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
