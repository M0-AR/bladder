from __future__ import annotations

import json

import pytest

from bladder_vision.annotations import (
    Annotation,
    build_voc_xml,
    clip_box,
    convert_vgg_to_voc,
    frame_filename,
    iter_annotations,
    load_vgg_metadata,
    parse_box,
)


def test_load_and_parse_via_annotation(tmp_path):
    payload = {
        "metadata": {
            "42": {
                "av": {"2": "NMIBC"},
                "z": [1.5],
                "xy": [0, 10, 20, 30, 40],
            }
        }
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    metadata = load_vgg_metadata(path)
    annotations = list(iter_annotations(metadata))
    assert len(annotations) == 1
    annotation = annotations[0]
    assert annotation.key == "42"
    assert annotation.label == "NMIBC"
    assert annotation.timestamps == (1.5,)
    assert parse_box(annotation.xy) == (10.0, 20.0, 30.0, 40.0)
    assert parse_box(annotation.xy, offset=0) == (0.0, 10.0, 20.0, 30.0)


def test_parse_box_rejects_invalid_size():
    with pytest.raises(ValueError):
        parse_box([0, 1, 2, 0, 4])
    with pytest.raises(ValueError):
        parse_box([0, 1])


def test_clip_box_handles_edges():
    assert clip_box((10, 20, 30, 40), 100, 100) == (10, 20, 40, 60)
    assert clip_box((-10, -10, 50, 50), 100, 100) == (0, 0, 40, 40)


def test_frame_filename_is_stable():
    assert frame_filename(1.5, 1) == "Frame_1_5_1.png"
    assert frame_filename(10, 3) == "Frame_10_3.png"


def test_build_voc_xml_contains_expected_fields():
    root = build_voc_xml(
        filename="frame.png",
        width=100,
        height=80,
        depth=3,
        xmin=1,
        ymin=2,
        xmax=30,
        ymax=40,
        class_name="Suspected_Region",
    )
    assert root.tag == "annotation"
    assert root.findtext("filename") == "frame.png"
    assert root.findtext("size/width") == "100"
    assert root.findtext("object/name") == "Suspected_Region"
    assert root.findtext("object/bndbox/xmax") == "30"


def test_convert_vgg_to_voc_writes_xml(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    images = tmp_path / "images"
    output = tmp_path / "annotations"
    images.mkdir()
    image_module.new("RGB", (100, 80), color=(10, 20, 30)).save(images / "Frame_1_5_1.png")
    payload = {
        "metadata": {
            "1": {"av": {"2": "NMIBC"}, "z": [1.5], "xy": [0, 10, 20, 30, 40]},
            "2": {"av": {"2": "Normal"}, "z": [2.0], "xy": []},
        }
    }
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    written = convert_vgg_to_voc(json_path=json_path, images_dir=images, output_dir=output)
    assert written == 1
    xml_path = output / "Frame_1_5_1.xml"
    assert xml_path.exists()
    assert "Suspected_Region" in xml_path.read_text(encoding="utf-8")


def test_annotation_is_frozen():
    annotation = Annotation("1", "Normal", (0.0,), (), {})
    with pytest.raises(AttributeError):
        annotation.label = "Ta"  # type: ignore[misc]
