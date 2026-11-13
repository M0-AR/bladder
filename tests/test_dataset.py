from __future__ import annotations

import json

import pytest

from bladder_vision.dataset import (
    SplitRatios,
    dataset_summary,
    pair_detection_files,
    split_classification_dataset,
    split_counts,
    split_detection_dataset,
)


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        SplitRatios(train=0.6, validation=0.2, test=0.1)
    with pytest.raises(ValueError):
        SplitRatios(train=0.0, validation=0.5, test=0.5)


def test_split_counts_preserve_total():
    counts = split_counts(10, SplitRatios())
    assert counts == {"train": 7, "validation": 2, "test": 1}
    assert sum(counts.values()) == 10
    with pytest.raises(ValueError):
        split_counts(2, SplitRatios())


def test_split_classification_dataset(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    for class_name in ("NMIBC", "Normal", "Ta"):
        directory = source / class_name
        directory.mkdir(parents=True)
        for index in range(10):
            (directory / f"{index}.png").write_text("x", encoding="utf-8")

    summary = split_classification_dataset(source=source, output=output, seed=7, copy=True)
    assert summary["NMIBC"] == {"train": 7, "validation": 2, "test": 1}
    assert len(list((output / "train" / "NMIBC").iterdir())) == 7
    assert len(list((output / "validation" / "Normal").iterdir())) == 2
    assert len(list((output / "test" / "Ta").iterdir())) == 1

    manifest = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 7
    assert manifest["counts"]["Normal"]["train"] == 7


def test_split_classification_is_reproducible(tmp_path):
    source = tmp_path / "source"
    for class_name in ("NMIBC", "Normal", "Ta"):
        directory = source / class_name
        directory.mkdir(parents=True)
        for index in range(10):
            (directory / f"{index}.png").write_text(str(index), encoding="utf-8")

    first = tmp_path / "first"
    second = tmp_path / "second"
    split_classification_dataset(source=source, output=first, seed=11)
    split_classification_dataset(source=source, output=second, seed=11)

    first_files = sorted(path.name for path in (first / "train" / "NMIBC").iterdir())
    second_files = sorted(path.name for path in (second / "train" / "NMIBC").iterdir())
    assert first_files == second_files


def test_split_detection_dataset(tmp_path):
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    for index in range(10):
        (images / f"frame_{index}.png").write_text("image", encoding="utf-8")
        (annotations / f"frame_{index}.txt").write_text("0,0,10,10\n", encoding="utf-8")

    pairs = pair_detection_files(images, annotations)
    assert len(pairs) == 10

    counts = split_detection_dataset(
        images_dir=images,
        annotations_dir=annotations,
        output=tmp_path / "output",
        seed=3,
    )
    assert counts == {"train": 7, "validation": 2, "test": 1}
    assert len(list((tmp_path / "output" / "train" / "images").iterdir())) == 7
    assert len(list((tmp_path / "output" / "test" / "annotations").iterdir())) == 1


def test_pair_detection_files_reports_missing(tmp_path):
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    (images / "frame.png").write_text("image", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        pair_detection_files(images, annotations)


def test_dataset_summary(tmp_path):
    for split in ("train", "validation", "test"):
        for class_name in ("NMIBC", "Normal"):
            directory = tmp_path / split / class_name
            directory.mkdir(parents=True)
            (directory / "a.png").write_text("x", encoding="utf-8")
    summary = dataset_summary(tmp_path)
    assert summary["train"]["NMIBC"] == 1
    assert summary["test"]["Normal"] == 1
