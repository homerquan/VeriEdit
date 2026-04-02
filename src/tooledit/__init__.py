from __future__ import annotations

from tooledit.schemas import EditRequest, EditResult
from tooledit.workflow import ToolEditWorkflow


def edit_image(
    source_image: str,
    prompt: str,
    reference_image: str | None = None,
    output_path: str | None = None,
    max_iterations: int = 3,
    preserve_metadata: bool = False,
    save_intermediates: bool = True,
    llm_model: str = "gemini-3-flash",
) -> EditResult:
    request = EditRequest(
        source_image=source_image,
        prompt=prompt,
        reference_image=reference_image,
        output_path=output_path,
        max_iterations=max_iterations,
        preserve_metadata=preserve_metadata,
        save_intermediates=save_intermediates,
        llm_model=llm_model,
    )
    return ToolEditWorkflow().run(request)


__all__ = ["EditRequest", "EditResult", "ToolEditWorkflow", "edit_image"]
