"""Reusable HippoSLAM components without project-specific training defaults.

Import individual modules explicitly so optional consumers do not load DMLab,
TorchVision, or a project's training entry point merely to access a utility.
"""
