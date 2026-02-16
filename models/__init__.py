"""
Models module for IvyFake detector
"""

from .detector import (
    IvyXDetector,
    SimplifiedIvyDetector,
    TemporalArtifactAnalyzer,
    SpatialArtifactAnalyzer,
    load_pretrained_ivydetector
)

from .multiscale_detector import (
    MultiScaleIvyDetector,
    TemporalPyramidExtractor,
    load_multiscale_detector
)

__all__ = [
    'IvyXDetector',
    'SimplifiedIvyDetector',
    'TemporalArtifactAnalyzer',
    'SpatialArtifactAnalyzer',
    'load_pretrained_ivydetector',
    'MultiScaleIvyDetector',
    'TemporalPyramidExtractor',
    'load_multiscale_detector'
]