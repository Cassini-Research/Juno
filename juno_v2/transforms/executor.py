from __future__ import annotations

from typing import Literal, TypedDict

from juno_v2.contracts.transforms import CustomTransformRecord
from juno_v2.transforms.catalog import get_builtin


class TransformExecutionMetadata(TypedDict, total=False):
    transform_id: str
    transform_source: Literal["builtin", "custom"]
    custom_name: str
    error: Literal["unknown_transform"]


def resolve_transform_instruction(
    *,
    transform_id: str,
    custom: CustomTransformRecord | None = None,
) -> tuple[str, TransformExecutionMetadata]:
    """Return model instruction and execution metadata for a transform id."""
    tid = (transform_id or "").strip()
    meta: TransformExecutionMetadata = {"transform_id": tid, "transform_source": "builtin"}
    builtin = get_builtin(tid)
    if custom is not None and custom.enabled:
        meta["transform_source"] = "custom"
        meta["custom_name"] = custom.name
        base = get_builtin(custom.base_transform_id) if custom.base_transform_id else None
        base_instr = (base.model_prompt_template if base else "") or ""
        instr = f"{base_instr}\nUser instruction: {custom.instruction}".strip() if custom.instruction else base_instr
        if not instr and builtin:
            instr = builtin.model_prompt_template
        return instr, meta
    if builtin is None:
        return "", {**meta, "error": "unknown_transform"}
    return builtin.model_prompt_template, meta
