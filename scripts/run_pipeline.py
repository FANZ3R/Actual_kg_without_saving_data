#!/usr/bin/env python3
"""
Main execution script for the Knowledge Graph Extraction Pipeline

Usage:
    python scripts/run_pipeline.py [--config CONFIG_PATH]

Examples:
    # Run with default configuration
    python scripts/run_pipeline.py

    # Run with custom configuration
    python scripts/run_pipeline.py --config config/custom.yaml
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import PipelineOrchestrator


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python scripts/run_pipeline.py

  # Run with custom configuration
  python scripts/run_pipeline.py --config config/production.yaml

  # Use environment variables for sensitive data
  export NEO4J_PASSWORD=your_password
  python scripts/run_pipeline.py
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Path to configuration file (default: config/default.yaml)'
    )

    args = parser.parse_args()

    # Print banner
    print("=" * 80)
    print("Fast Knowledge Graph Extraction Pipeline")
    print("=" * 80)
    print()

    try:
        # Initialize orchestrator
        print(f"Loading configuration from: {args.config}")
        orchestrator = PipelineOrchestrator(config_path=args.config)

        # Initialize components
        print("Initializing pipeline components...")
        orchestrator.initialize_components()

        # Run pipeline
        print("Starting pipeline execution...")
        print()
        result = orchestrator.run()

        print()
        print("=" * 80)
        print("Pipeline execution completed successfully!")
        print("=" * 80)

        return 0

    except KeyboardInterrupt:
        print("\n\nPipeline execution interrupted by user")
        return 130

    except Exception as e:
        print(f"\n\nERROR: Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())