"""Command line processing and evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from queuesense import __version__
from queuesense.config import load_settings
from queuesense.evaluation import evaluate_run, format_report, load_ground_truth
from queuesense.pipeline import execute_run
from queuesense.storage import RunStore
from queuesense.zones import load_zone_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="queuesense",
        description="Measure one queue at one service counter in a fixed-camera video.",
    )
    parser.add_argument("--version", action="version", version=f"QueueSense {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    process = commands.add_parser("process", help="Analyze one video and store a run.")
    process.add_argument("--video", required=True, type=Path, help="Input video file.")
    process.add_argument("--zones", required=True, type=Path, help="Normalized queue and service polygons.")
    process.add_argument("--config", type=Path, default=None, help="Settings JSON. Defaults to configs/default.json.")
    process.add_argument("--model", default=None, help="Model file or Ultralytics model name, such as yolo26n.pt.")
    process.add_argument("--conf", type=float, default=None, help="Person-detection confidence.")
    process.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    process.add_argument("--stride", type=int, default=None, help="Process every Nth frame. Default is 1.")
    process.add_argument("--db", type=Path, default=None, help="SQLite database path.")
    process.add_argument("--output-root", type=Path, default=None, help="Directory that will contain the run folder.")

    evaluate = commands.add_parser("evaluate", help="Score one stored run against manual annotations.")
    evaluate.add_argument("--db", required=True, type=Path)
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--ground-truth", required=True, type=Path)
    evaluate.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    return parser


def _process(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    if args.model:
        settings.model_path = args.model
    if args.conf is not None:
        settings.confidence = args.conf
    if args.device:
        settings.device = args.device
    if args.stride is not None:
        settings.frame_stride = args.stride
    settings.validate()
    queue, service = load_zone_file(args.zones)
    from queuesense.detector import PersonDetector

    detector = PersonDetector(settings.model_path, settings.device, settings.image_size)

    def _progress(fraction: float, message: str) -> None:
        print(f"[{fraction:6.1%}] {message}", flush=True)

    result = execute_run(
        args.video,
        queue,
        service,
        settings,
        detector,
        output_root=args.output_root,
        db_path=args.db,
        progress=_progress,
    )
    summary = result.summary
    print(f"Run {result.run_id} stored with status {result.status}.")
    print(f"Annotated video: {result.annotated_video}")
    print(f"Output directory: {result.output_dir}")
    print(
        "Outcomes: "
        f"completed={summary['completed']}, abandoned={summary['abandoned']}, "
        f"lost={summary['lost']}, incomplete={summary['incomplete']}"
    )
    if summary["mean_waiting_sec"] is None:
        print("Mean completed wait: unavailable (no completed visits with a measured wait).")
    else:
        print(f"Mean completed wait: {summary['mean_waiting_sec']:.2f} s")
    print(f"Last prediction status: {summary['last_prediction_status']}")
    if result.processing_fps is None:
        print("Processing FPS: unavailable")
    else:
        print(f"Processing FPS: {result.processing_fps:.2f}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    store = RunStore(args.db)
    run = store.get_run(args.run_id)
    if run is None:
        raise RuntimeError(f"No run found with id {args.run_id}")
    hardware = json.loads(run["hardware_json"]) if run.get("hardware_json") else {}
    report = evaluate_run(
        store.get_metrics(args.run_id),
        store.get_visits(args.run_id),
        load_ground_truth(args.ground_truth),
        {
            "status": run["status"],
            "processing_fps": run["processing_fps"],
            "hardware": hardware,
        },
    )
    text = format_report(report)
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "process":
            return _process(args)
        if args.command == "evaluate":
            return _evaluate(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
