"""VGG Image Annotator (VIA) JSON parsing and Pascal VOC XML export."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from .constants import DETECTION_LABEL


@dataclass(frozen=True)
class Annotation:
    """One VIA annotation record."""

    key: str
    label: str
    timestamps: tuple[float, ...]
    xy: tuple[float, ...]
    raw: Mapping[str, Any]


def load_vgg_metadata(path: str | Path) -> dict[str, Any]:
    """Load the ``metadata`` object from a VIA JSON export."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} does not contain a VIA 'metadata' object")
    return metadata


def parse_annotation(key: str, info: Mapping[str, Any]) -> Annotation:
    """Convert one raw VIA record into a typed :class:`Annotation`."""
    av = info.get("av") or {}
    label = str(av.get("2", "")).strip()
    timestamps = tuple(float(value) for value in (info.get("z") or []))
    xy = tuple(float(value) for value in (info.get("xy") or []))
    return Annotation(key=str(key), label=label, timestamps=timestamps, xy=xy, raw=info)


def iter_annotations(metadata: Mapping[str, Any]) -> Iterator[Annotation]:
    """Yield annotations in the order stored by VIA."""
    for key, info in metadata.items():
        if isinstance(info, Mapping):
            yield parse_annotation(key, info)


def parse_box(xy: Iterable[float], offset: int = 1) -> tuple[float, float, float, float]:
    """Return ``(x, y, width, height)`` from a VIA coordinate list.

    The original research export stores a leading sentinel before the four box
    values, so ``offset`` defaults to ``1``. Pass ``offset=0`` for plain
    ``[x, y, width, height]`` lists.
    """
    values = list(xy)[offset:]
    if len(values) < 4:
        raise ValueError(f"Expected at least four box values, got {len(values)}")
    x, y, width, height = values[:4]
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid box size: width={width}, height={height}")
    return x, y, width, height


def clip_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    """Clip an ``(x, y, w, h)`` box to image bounds and return integer x1/y1/x2/y2."""
    x, y, box_width, box_height = box
    x1 = max(0, min(int(round(x)), width))
    y1 = max(0, min(int(round(y)), height))
    x2 = max(0, min(int(round(x + box_width)), width))
    y2 = max(0, min(int(round(y + box_height)), height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Box falls outside the image")
    return x1, y1, x2, y2


def build_voc_xml(
    *,
    filename: str,
    width: int,
    height: int,
    depth: int,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    class_name: str = DETECTION_LABEL,
    folder: str = "images",
) -> ET.Element:
    """Build a Pascal VOC ``<annotation>`` tree for one bounding box."""
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = folder
    ET.SubElement(root, "filename").text = filename
    ET.SubElement(root, "path").text = f"{folder}/{filename}"

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Custom Dataset"

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(root, "segmented").text = "0"

    obj = ET.SubElement(root, "object")
    ET.SubElement(obj, "name").text = class_name
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = "0"
    ET.SubElement(obj, "difficult").text = "0"

    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(xmin)
    ET.SubElement(bndbox, "ymin").text = str(ymin)
    ET.SubElement(bndbox, "xmax").text = str(xmax)
    ET.SubElement(bndbox, "ymax").text = str(ymax)
    return root


def write_voc_xml(
    *,
    filename: str,
    width: int,
    height: int,
    depth: int,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    output_path: str | Path,
    class_name: str = DETECTION_LABEL,
) -> Path:
    """Write a Pascal VOC XML file and return its path."""
    root = build_voc_xml(
        filename=filename,
        width=width,
        height=height,
        depth=depth,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        class_name=class_name,
    )
    ET.indent(root, space="    ")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def frame_filename(timestamp: float, index: int) -> str:
    """Canonical frame name used by extraction and annotation conversion."""
    stamp = f"{timestamp:g}".replace(".", "_")
    return f"Frame_{stamp}_{index}.png"


def convert_vgg_to_voc(
    *,
    json_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    xy_offset: int = 1,
    class_name: str = DETECTION_LABEL,
) -> int:
    """Convert every point annotation in a VIA export to a Pascal VOC XML file.

    Frames are matched by the canonical name produced by :func:`frame_filename`.
    Annotations without exactly one timestamp or without a box are skipped.
    Returns the number of XML files written.
    """
    from PIL import Image

    metadata = load_vgg_metadata(json_path)
    images = Path(images_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written = 0

    for index, annotation in enumerate(iter_annotations(metadata), start=1):
        if len(annotation.timestamps) != 1 or not annotation.xy:
            continue
        name = frame_filename(annotation.timestamps[0], index)
        image_path = images / name
        if not image_path.exists():
            continue
        try:
            box = parse_box(annotation.xy, offset=xy_offset)
        except ValueError:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
            depth = len(image.getbands())
        x1, y1, x2, y2 = clip_box(box, width, height)
        write_voc_xml(
            filename=name,
            width=width,
            height=height,
            depth=depth,
            xmin=x1,
            ymin=y1,
            xmax=x2,
            ymax=y2,
            output_path=output / f"{Path(name).stem}.xml",
            class_name=class_name,
        )
        written += 1
    return written
