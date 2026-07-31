"""Unified image indexing and sensor analysis for heat-pump frost data."""

__version__ = "0.1.0"

from frost_analysis import pipeline as _pipeline
from frost_analysis.analysis import analyze
from frost_analysis.prepare import prepare
from frost_analysis.process import process

run_pipeline = _pipeline.run_pipeline

__all__ = ["prepare", "process", "analyze", "run_pipeline"]
