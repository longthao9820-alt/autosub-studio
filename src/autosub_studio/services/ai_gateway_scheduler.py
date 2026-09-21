"""Shared AI Gateway scheduler for subtitle extraction and translation.

The implementation remains in ``ai_ocr_scheduler`` for backward compatibility
with existing projects/tests. Both public accessors point at the same process-
wide scheduler, so ``sub`` and ``prime`` share bounded concurrency and fair job
rotation instead of creating independent request storms.
"""

from __future__ import annotations

from .ai_ocr_scheduler import (
    AIOCRScheduler,
    get_global_scheduler,
    shutdown_global_scheduler,
)

AIGatewayScheduler = AIOCRScheduler
get_global_gateway_scheduler = get_global_scheduler
shutdown_global_gateway_scheduler = shutdown_global_scheduler

__all__ = [
    "AIGatewayScheduler",
    "get_global_gateway_scheduler",
    "shutdown_global_gateway_scheduler",
]

