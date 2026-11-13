"""Reproducible dataset splitting and inventory helpers."""

from __future__ import annotations

import json
import random
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from .constants import DEFAULT_SEED

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ANNOTATION_SUFFIXES = {".txt", ".xml"}


@dataclass(frozen=True)
class SplitRatios:
    """Train/validation/test proportions. They must sum to 1."""

    train: float = 0.7
    validation: float = 0.2
    test: float = 0.1

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(value <= 0 for value in values):
            raise ValueError("Every split ratio must be positive")
        if abs(sum(values) - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1, got {sum(values):.4f}")


def split_counts(total: int, ratios: SplitRatios) -> dict[str, int]:
    """Return integer counts that preserve the requested proportions."""
    if total < 3:
        raise ValueError("At least three items are required to create three splits")
    train = int(total * ratios.train)
    validation = int(total * ratios.validation)
    test = total - train - validation
    if test == 0:
        test = 1
        if train > validation:
            train -= 1
        else:
            validation -= 1
    return {"train": train, "validation": validation, "test": test}


def _copy_or_move(source: Path, destination: Path, copy: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(source, destination)
    else:
        shutil.move(str(source), str(destination))


def split_classification_dataset(
    *,
    source: str | Path,
    output: str | Path,
    ratios: SplitRatios | None = None,
    seed: int = DEFAULT_SEED,
    copy: bool = True,
    manifest: bool = True,
) -> dict[str, dict[str, int]]:
    """Create stratified ``train``/``validation``/``test`` class folders.

    ``source`` must contain one directory per class. Files are shuffled inside each
    class with a seeded RNG, so the split is reproducible. Returns per-class counts.
    """
    resolved_ratios = ratios or SplitRatios()
    source_path = Path(source)
    output_path = Path(output)
    classes = sorted(path.name for path in source_path.iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"No class directories found in {source_path}")

    rng = random.Random(seed)
    summary: dict[str, dict[str, int]] = {}

    for class_name in classes:
        files = sorted(path for path in (source_path / class_name).iterdir() if path.is_file())
        counts = split_counts(len(files), resolved_ratios)
        rng.shuffle(files)
        cursor = 0
        for split_name, count in counts.items():
            selected = files[cursor : cursor + count]
            cursor += count
            destination = output_path / split_name / class_name
            for file_path in selected:
                _copy_or_move(file_path, destination / file_path.name, copy)
        summary[class_name] = counts

    if manifest:
        payload = {
            "kind": "classification",
            "source": str(source_path),
            "seed": seed,
            "ratios": asdict(resolved_ratios),
            "counts": summary,
        }
        (output_path).mkdir(parents=True, exist_ok=True)
        (output_path / "split_manifest.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return summary


def pair_detection_files(
    images_dir: str | Path,
    annotations_dir: str | Path,
    annotation_suffix: str | None = None,
) -> list[tuple[Path, Path]]:
    """Pair images with annotation files that share the same stem."""
    images_path = Path(images_dir)
    annotations_path = Path(annotations_dir)
    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []

    for image_path in sorted(images_path.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if annotation_suffix:
            annotation_path = annotations_path / f"{image_path.stem}{annotation_suffix}"
        else:
            candidates = [
                annotations_path / f"{image_path.stem}{suffix}"
                for suffix in sorted(ANNOTATION_SUFFIXES)
            ]
            annotation_path = next((path for path in candidates if path.exists()), candidates[0])
        if not annotation_path.exists():
            missing.append(image_path.name)
            continue
        pairs.append((image_path, annotation_path))

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing annotations for {len(missing)} images: {preview}")
    if not pairs:
        raise ValueError("No image/annotation pairs found")
    return pairs


def split_detection_dataset(
    *,
    images_dir: str | Path,
    annotations_dir: str | Path,
    output: str | Path,
    ratios: SplitRatios | None = None,
    seed: int = DEFAULT_SEED,
    annotation_suffix: str | None = None,
    copy: bool = True,
    manifest: bool = True,
) -> dict[str, int]:
    """Split an object-detection dataset while keeping image/annotation pairs together."""
    resolved_ratios = ratios or SplitRatios()
    pairs = pair_detection_files(images_dir, annotations_dir, annotation_suffix)
    counts = split_counts(len(pairs), resolved_ratios)
    rng = random.Random(seed)
    rng.shuffle(pairs)

    output_path = Path(output)
    cursor = 0
    for split_name, count in counts.items():
        selected = pairs[cursor : cursor + count]
        cursor += count
        for image_path, annotation_path in selected:
            _copy_or_move(image_path, output_path / split_name / "images" / image_path.name, copy)
            _copy_or_move(
                annotation_path,
                output_path / split_name / "annotations" / annotation_path.name,
                copy,
            )

    if manifest:
        payload = {
            "kind": "detection",
            "images": str(images_dir),
            "annotations": str(annotations_dir),
            "seed": seed,
            "ratios": asdict(resolved_ratios),
            "counts": counts,
        }
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "split_manifest.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return counts


def dataset_summary(root: str | Path) -> dict[str, dict[str, int]]:
    """Count files per split and class in a classification dataset directory."""
    root_path = Path(root)
    summary: dict[str, dict[str, int]] = {}
    for split_path in sorted(path for path in root_path.iterdir() if path.is_dir()):
        if split_path.name not in {"train", "validation", "test"}:
            continue
        summary[split_path.name] = {}
        for class_path in sorted(path for path in split_path.iterdir() if path.is_dir()):
            summary[split_path.name][class_path.name] = sum(
                1 for path in class_path.iterdir() if path.is_file()
            )
    return summary


def iter_image_files(directory: str | Path) -> Iterable[Path]:
    """Yield image files in a directory in deterministic order."""
    for path in sorted(Path(directory).iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path
