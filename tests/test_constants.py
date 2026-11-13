from __future__ import annotations

from bladder_vision.constants import CLASS_DESCRIPTIONS, CLASS_NAMES, DETECTION_LABEL


def test_class_order_is_canonical():
    assert CLASS_NAMES == ("NMIBC", "Normal", "Ta")
    assert set(CLASS_DESCRIPTIONS) == set(CLASS_NAMES)
    assert DETECTION_LABEL == "Suspected_Region"
