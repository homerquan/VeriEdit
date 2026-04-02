from __future__ import annotations

import argparse
from pathlib import Path

from veriedit.manual_eval import build_manual_eval_from_run, build_manual_eval_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a single markdown manual-eval file.")
    parser.add_argument("--run-id", help="Existing run id under the artifact root.")
    parser.add_argument("--artifact-root", default="runs", help="Artifact root for --run-id mode.")
    parser.add_argument("--source", help="Source image path.")
    parser.add_argument("--reference", help="Reference image path.")
    parser.add_argument("--result", help="Result image path.")
    parser.add_argument("--report-json", help="Optional report.json path.")
    parser.add_argument("--observation-json", help="Optional observation_trace.json path.")
    parser.add_argument("--prompt", help="Optional prompt override.")
    parser.add_argument("--title", help="Optional markdown title.")
    parser.add_argument("--link-images", action="store_true", help="Use relative file links instead of embedding images.")
    parser.add_argument("--output", required=True, help="Output markdown path.")
    args = parser.parse_args()

    if args.run_id:
        build_manual_eval_from_run(
            args.run_id,
            artifact_root=args.artifact_root,
            output_path=args.output,
            embed_images=not args.link_images,
        )
        return

    if not args.source or not args.result:
        parser.error("Either --run-id or both --source and --result are required.")

    build_manual_eval_markdown(
        source_image=args.source,
        reference_image=args.reference,
        result_image=args.result,
        report_json=args.report_json,
        observation_json=args.observation_json,
        prompt=args.prompt,
        title=args.title,
        output_path=args.output,
        embed_images=not args.link_images,
    )


if __name__ == "__main__":
    main()
