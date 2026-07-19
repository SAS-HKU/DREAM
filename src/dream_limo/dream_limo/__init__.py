"""Occlusion-aware DREAM deployment package for the AgileX LIMO."""

from .limo_scale import DeploymentConfig, default_deployment_config

__all__ = ["DeploymentConfig", "default_deployment_config"]
__version__ = "0.1.0"
