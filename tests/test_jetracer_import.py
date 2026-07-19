"""The JetRacer backend must be *importable* on a plain PC (no Jetson libs);
only instantiating JetRacerController may require the hardware. This guards the
lazy-import contract so tests and the mock CLI stay hardware-free."""
import importlib

import pytest


def test_jetracer_module_imports_without_hardware():
    module = importlib.import_module("aliengo.robot.jetracer")
    assert hasattr(module, "JetRacerController")


def test_detector_module_imports_without_hardware():
    module = importlib.import_module("aliengo.vision.detector")
    assert hasattr(module, "JetsonDetector")
    # The Detection value type is pure Python and usable off-Jetson.
    det = module.Detection(
        label="bottle", confidence=0.9, cx=0.2, cy=0.5, width=0.1, height=0.3
    )
    assert det.position == "left"
    assert det.area == pytest.approx(0.03)
