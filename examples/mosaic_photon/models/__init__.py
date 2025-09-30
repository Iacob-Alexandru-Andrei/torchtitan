"""Composer-based model definitions for Mosaic Photon examples."""

from __future__ import annotations

from .mpt_mup import (
    HAS_MPT_MUP_SUPPORT,
    MPT_MUP_CONFIGS,
    MPTMuPModelArgs,
    TitanComposerMPTMuP,
)

__all__ = [
    "HAS_MPT_MUP_SUPPORT",
    "MPT_MUP_CONFIGS",
    "MPTMuPModelArgs",
    "TitanComposerMPTMuP",
]
