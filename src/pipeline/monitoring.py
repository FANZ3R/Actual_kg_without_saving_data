"""
Pipeline Monitoring Module
Tracks pipeline progress, metrics, and performance
"""

import logging
import time
from typing import Dict, Any
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class PipelineMonitor:
    """Monitors pipeline execution and collects metrics"""

    def __init__(self):
        """Initialize PipelineMonitor"""
        self.start_time = None
        self.end_time = None
        self.stage_times = {}
        self.stage_start_times = {}
        self.metrics = defaultdict(dict)
        self.errors = []
        self.warnings = []
        self.logger = logger

    def start_pipeline(self):
        """Mark pipeline start"""
        self.start_time = time.time()
        self.logger.info("Pipeline execution started")

    def end_pipeline(self):
        """Mark pipeline end"""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        self.logger.info(f"Pipeline execution completed in {duration:.2f} seconds")

    def start_stage(self, stage_name: str):
        """Mark stage start"""
        self.stage_start_times[stage_name] = time.time()
        self.logger.info(f"Stage '{stage_name}' started")

    def end_stage(self, stage_name: str):
        """Mark stage end"""
        if stage_name in self.stage_start_times:
            start_time = self.stage_start_times[stage_name]
            duration = time.time() - start_time
            self.stage_times[stage_name] = duration
            self.logger.info(f"Stage '{stage_name}' completed in {duration:.2f} seconds")
        else:
            self.logger.warning(f"Stage '{stage_name}' end called without start")

    def record_metric(self, stage: str, metric_name: str, value: Any):
        """Record a metric for a stage"""
        self.metrics[stage][metric_name] = value

    def record_error(self, stage: str, error: str):
        """Record an error"""
        error_record = {
            'stage': stage,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        self.errors.append(error_record)
        self.logger.error(f"Error in {stage}: {error}")

    def record_warning(self, stage: str, warning: str):
        """Record a warning"""
        warning_record = {
            'stage': stage,
            'warning': warning,
            'timestamp': datetime.now().isoformat()
        }
        self.warnings.append(warning_record)
        self.logger.warning(f"Warning in {stage}: {warning}")

    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline execution summary"""
        total_duration = self.end_time - self.start_time if self.end_time else time.time() - self.start_time

        summary = {
            'start_time': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            'total_duration_seconds': total_duration,
            'stage_durations': self.stage_times,
            'metrics': dict(self.metrics),
            'errors_count': len(self.errors),
            'warnings_count': len(self.warnings),
            'errors': self.errors,
            'warnings': self.warnings
        }

        return summary

    def print_summary(self):
        """Print a formatted summary to console"""
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Total Duration: {summary['total_duration_seconds']:.2f} seconds")
        print()

        print("Stage Durations:")
        for stage, duration in summary['stage_durations'].items():
            print(f"  {stage}: {duration:.2f}s")
        print()

        print("Metrics:")
        for stage, metrics in summary['metrics'].items():
            print(f"  {stage}:")
            for metric_name, value in metrics.items():
                print(f"    {metric_name}: {value}")
        print()

        if summary['errors_count'] > 0:
            print(f"Errors: {summary['errors_count']}")
            for error in summary['errors'][:5]:
                print(f"  [{error['stage']}] {error['error']}")
            if summary['errors_count'] > 5:
                print(f"  ... and {summary['errors_count'] - 5} more errors")
            print()

        if summary['warnings_count'] > 0:
            print(f"Warnings: {summary['warnings_count']}")
            for warning in summary['warnings'][:5]:
                print(f"  [{warning['stage']}] {warning['warning']}")
            if summary['warnings_count'] > 5:
                print(f"  ... and {summary['warnings_count'] - 5} more warnings")
            print()

        print("=" * 80 + "\n")