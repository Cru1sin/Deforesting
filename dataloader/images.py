"""Dataset image collection and coverage operations."""

from src.frost_analysis.dataset.images import (
    RGB_CAMERA_ORDER,
    build_rgb_coverage_intervals,
    collect_cycle_images,
    copy_image,
    image_coverage_intervals,
    image_metadata_frame,
    materialize_cycle_image_members,
    materialize_cycle_images,
    rgb_overall_intervals,
    rgb_stage_metrics,
    scan_cycle_images,
    summarize_rgb_coverage,
)

__all__ = [
    "RGB_CAMERA_ORDER",
    "build_rgb_coverage_intervals",
    "collect_cycle_images",
    "copy_image",
    "image_coverage_intervals",
    "image_metadata_frame",
    "materialize_cycle_image_members",
    "materialize_cycle_images",
    "rgb_overall_intervals",
    "rgb_stage_metrics",
    "scan_cycle_images",
    "summarize_rgb_coverage",
]
