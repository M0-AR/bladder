"""Command-line interface for the Bladder Vision pipeline."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

from . import __version__
from .annotations import convert_vgg_to_voc, load_vgg_metadata
from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONFIDENCE,
    DEFAULT_EPOCHS,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DETECTOR_FILE,
    LABELMAP_FILE,
)
from .dataset import (
    SplitRatios,
    dataset_summary,
    split_classification_dataset,
    split_detection_dataset,
)
from .metrics import binary_screening_metrics, classification_report, format_report


def _repo_root() -> Path:
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "models").is_dir():
        return candidate
    return Path.cwd()


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def cmd_version(_: argparse.Namespace) -> int:
    print(f"bladder-vision {__version__}")
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    print("Bladder Vision environment")
    print("-" * 40)
    print(f"Python           {sys.version.split()[0]}")
    for module, label in (
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("tensorflow", "TensorFlow"),
        ("matplotlib", "Matplotlib"),
    ):
        print(f"{label:<16} {'ok' if _module_available(module) else 'missing'}")

    roots = {Path.cwd(), _repo_root()}
    print("-" * 40)
    for root in sorted(roots):
        detector = root / "models" / DETECTOR_FILE
        labels = root / "models" / LABELMAP_FILE
        if detector.exists():
            size_mb = detector.stat().st_size / (1024 * 1024)
            print(f"Detector         {detector} ({size_mb:.1f} MB)")
        if labels.exists():
            print(f"Label map        {labels}")
    print("-" * 40)
    print("Core pipeline dependencies are optional; install the extras you need.")
    return 0


def cmd_convert_annotations(args: argparse.Namespace) -> int:
    written = convert_vgg_to_voc(
        json_path=args.json,
        images_dir=args.images,
        output_dir=args.output,
        xy_offset=args.xy_offset,
        class_name=args.class_name,
    )
    print(f"Wrote {written} Pascal VOC XML file(s) to {args.output}")
    return 0


def cmd_extract_frames(args: argparse.Namespace) -> int:
    from .frames import extract_annotated_frames, save_frames

    metadata = load_vgg_metadata(args.json)
    frames = extract_annotated_frames(
        video_path=args.video,
        metadata=metadata,
        crop=args.crop,
        every_nth_annotation=args.every,
        xy_offset=args.xy_offset,
    )
    if not frames:
        print("No frames were extracted. Check the annotations and video timestamp units.")
        return 1
    written = save_frames(
        frames,
        output_dir=args.output,
        group_by_label=not args.flat,
        annotations_dir=args.annotations_dir,
    )
    print(f"Wrote {written} frame(s) to {args.output}")
    return 0


def _ratios(args: argparse.Namespace) -> SplitRatios:
    return SplitRatios(train=args.train, validation=args.validation, test=args.test)


def cmd_split_classification(args: argparse.Namespace) -> int:
    summary = split_classification_dataset(
        source=args.source,
        output=args.output,
        ratios=_ratios(args),
        seed=args.seed,
        copy=not args.move,
    )
    print(json.dumps(summary, indent=2))
    return 0


def cmd_split_detection(args: argparse.Namespace) -> int:
    counts = split_detection_dataset(
        images_dir=args.images,
        annotations_dir=args.annotations,
        output=args.output,
        ratios=_ratios(args),
        seed=args.seed,
        annotation_suffix=args.annotation_suffix,
        copy=not args.move,
    )
    print(json.dumps(counts, indent=2))
    return 0


def cmd_train_classifier(args: argparse.Namespace) -> int:
    from .classifier import train_classifier

    result = train_classifier(
        data_dir=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        learning_rate=args.learning_rate,
        seed=args.seed,
        patience=args.patience,
        steps_per_epoch=args.steps,
    )
    print(f"Model: {result['model_path']}")
    print(f"Classes: {', '.join(result['class_names'])}")
    print(f"Final training accuracy: {result['history']['accuracy'][-1]:.4f}")
    return 0


def cmd_evaluate_classifier(args: argparse.Namespace) -> int:
    from .classifier import evaluate_classifier

    report = evaluate_classifier(
        model_path=args.model,
        data_dir=args.data,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
    print(format_report(report))
    return 0


def cmd_classify_video(args: argparse.Namespace) -> int:
    from .video import classify_video

    predictions = classify_video(
        video_path=args.video,
        model_path=args.model,
        output_path=args.output,
        start_seconds=args.start,
        every_n=args.every,
        display=args.display,
        max_frames=args.max_frames,
        input_size=(args.width, args.height),
    )
    print(f"Classified {len(predictions)} sampled frame(s)")
    if predictions:
        last = predictions[-1]
        print(f"Last frame: {last.label} ({last.score:.3f})")
    return 0


def cmd_detect_video(args: argparse.Namespace) -> int:
    from .detector import TFLiteDetector, detect_video

    detector = TFLiteDetector(
        model_path=args.model,
        labels_path=args.labels,
        threshold=args.threshold,
    )
    processed = detect_video(
        video_path=args.video,
        detector=detector,
        output_path=args.output,
        display=args.display,
        every_n=args.every,
        max_frames=args.max_frames,
    )
    print(f"Processed {processed} frame(s)")
    return 0


def cmd_panorama(args: argparse.Namespace) -> int:
    from .panorama import save_panorama, stitch_video

    panorama = stitch_video(
        video_path=args.video,
        every_n=args.every,
        max_frames=args.max_frames,
    )
    destination = save_panorama(panorama, args.output)
    print(f"Panorama written to {destination}")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    print(json.dumps(dataset_summary(args.data), indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    with Path(args.csv).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print("The CSV file is empty.")
        return 1
    y_true = [row[args.true_column] for row in rows]
    if args.score_column:
        truths = [value == args.positive_label for value in y_true]
        scores = [float(row[args.score_column]) for row in rows]
        metrics = binary_screening_metrics(truths, scores, threshold=args.threshold)
        print(json.dumps(metrics, indent=2))
        return 0
    y_pred = [row[args.pred_column] for row in rows]
    report = classification_report(y_true, y_pred)
    print(format_report(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bladder-vision",
        description="Research-grade cystoscopy AI pipeline: annotations, training, detection and evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    version = subparsers.add_parser("version", help="Print the package version")
    version.set_defaults(func=cmd_version)

    doctor = subparsers.add_parser("doctor", help="Check optional dependencies and model files")
    doctor.set_defaults(func=cmd_doctor)

    convert = subparsers.add_parser(
        "convert-annotations", help="Convert VIA JSON annotations to Pascal VOC XML"
    )
    convert.add_argument("--json", required=True, help="VIA JSON export")
    convert.add_argument("--images", required=True, help="Directory containing extracted frames")
    convert.add_argument("--output", required=True, help="Directory for XML annotations")
    convert.add_argument("--xy-offset", type=int, default=1, help="Offset of box values")
    convert.add_argument("--class-name", default="Suspected_Region", help="VOC object name")
    convert.set_defaults(func=cmd_convert_annotations)

    extract = subparsers.add_parser("extract-frames", help="Extract annotated frames from a video")
    extract.add_argument("--video", required=True)
    extract.add_argument("--json", required=True, help="VIA JSON export")
    extract.add_argument("--output", required=True)
    extract.add_argument("--annotations-dir", help="Optional directory for VOC XML labels")
    extract.add_argument("--every", type=int, default=1, help="Use every Nth annotation")
    extract.add_argument("--xy-offset", type=int, default=1)
    extract.add_argument("--crop", action="store_true", help="Save lesion crops")
    extract.add_argument("--flat", action="store_true", help="Do not group output by class")
    extract.set_defaults(func=cmd_extract_frames)

    split_cls = subparsers.add_parser(
        "split-classification", help="Create stratified train/validation/test class folders"
    )
    split_cls.add_argument("--source", required=True)
    split_cls.add_argument("--output", required=True)
    split_cls.add_argument("--train", type=float, default=0.7)
    split_cls.add_argument("--validation", type=float, default=0.2)
    split_cls.add_argument("--test", type=float, default=0.1)
    split_cls.add_argument("--seed", type=int, default=DEFAULT_SEED)
    split_cls.add_argument("--move", action="store_true", help="Move instead of copy")
    split_cls.set_defaults(func=cmd_split_classification)

    split_det = subparsers.add_parser(
        "split-detection", help="Create paired train/validation/test detection folders"
    )
    split_det.add_argument("--images", required=True)
    split_det.add_argument("--annotations", required=True)
    split_det.add_argument("--output", required=True)
    split_det.add_argument("--train", type=float, default=0.7)
    split_det.add_argument("--validation", type=float, default=0.2)
    split_det.add_argument("--test", type=float, default=0.1)
    split_det.add_argument("--seed", type=int, default=DEFAULT_SEED)
    split_det.add_argument("--annotation-suffix", help="e.g. .xml or .txt")
    split_det.add_argument("--move", action="store_true")
    split_det.set_defaults(func=cmd_split_detection)

    train = subparsers.add_parser("train-classifier", help="Train the frame classifier")
    train.add_argument("--data", required=True, help="Dataset with train/ and optional validation/")
    train.add_argument("--output", default="bladder_types.keras")
    train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    train.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    train.add_argument("--height", type=int, default=DEFAULT_INPUT_HEIGHT)
    train.add_argument("--width", type=int, default=DEFAULT_INPUT_WIDTH)
    train.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--steps", type=int, help="Optional steps per epoch")
    train.set_defaults(func=cmd_train_classifier)

    evaluate_classifier = subparsers.add_parser(
        "evaluate-classifier", help="Evaluate a trained model on data/test"
    )
    evaluate_classifier.add_argument("--model", required=True)
    evaluate_classifier.add_argument("--data", required=True)
    evaluate_classifier.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    evaluate_classifier.add_argument("--height", type=int, default=DEFAULT_INPUT_HEIGHT)
    evaluate_classifier.add_argument("--width", type=int, default=DEFAULT_INPUT_WIDTH)
    evaluate_classifier.add_argument("--seed", type=int, default=DEFAULT_SEED)
    evaluate_classifier.set_defaults(func=cmd_evaluate_classifier)

    classify = subparsers.add_parser(
        "classify-video", help="Overlay classifier predictions on video"
    )
    classify.add_argument("--video", required=True)
    classify.add_argument("--model", required=True)
    classify.add_argument("--output", help="Annotated MP4 path")
    classify.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    classify.add_argument("--every", type=int, default=1, help="Run inference every N frames")
    classify.add_argument("--max-frames", type=int)
    classify.add_argument("--height", type=int, default=DEFAULT_INPUT_HEIGHT)
    classify.add_argument("--width", type=int, default=DEFAULT_INPUT_WIDTH)
    classify.add_argument("--display", action="store_true")
    classify.set_defaults(func=cmd_classify_video)

    detect = subparsers.add_parser("detect-video", help="Run TFLite detection on video")
    detect.add_argument("--video", required=True)
    detect.add_argument("--model", default="models/detect.tflite")
    detect.add_argument("--labels", default="models/labelmap.txt")
    detect.add_argument("--output", help="Annotated MP4 path")
    detect.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE)
    detect.add_argument("--every", type=int, default=1)
    detect.add_argument("--max-frames", type=int)
    detect.add_argument("--display", action="store_true")
    detect.set_defaults(func=cmd_detect_video)

    panorama = subparsers.add_parser("panorama", help="Stitch sampled video frames")
    panorama.add_argument("--video", required=True)
    panorama.add_argument("--output", default="panorama.png")
    panorama.add_argument("--every", type=int, default=30)
    panorama.add_argument("--max-frames", type=int, default=60)
    panorama.set_defaults(func=cmd_panorama)

    summary = subparsers.add_parser("summary", help="Count images in a classification dataset")
    summary.add_argument("--data", required=True)
    summary.set_defaults(func=cmd_summary)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate predictions stored in a CSV")
    evaluate.add_argument("--csv", required=True)
    evaluate.add_argument("--true-column", default="y_true")
    evaluate.add_argument("--pred-column", default="y_pred")
    evaluate.add_argument("--score-column", help="Use binary screening metrics with this score")
    evaluate.add_argument("--positive-label", default="1")
    evaluate.add_argument("--threshold", type=float, default=0.5)
    evaluate.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
