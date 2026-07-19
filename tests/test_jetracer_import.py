"""The JetRacer backend must be *importable* on a plain PC (no Jetson libs);
only instantiating JetRacerController may require the hardware. This guards the
lazy-import contract so tests and the mock CLI stay hardware-free."""
import importlib
import threading
import time

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


def test_jetracer_estop_interrupts_timed_motion_without_hardware():
    module = importlib.import_module("aliengo.robot.jetracer")

    class FakeCar:
        throttle = 0.0
        steering = 0.0

    controller = object.__new__(module.JetRacerController)
    controller.car = FakeCar()
    controller.detector = None
    controller._following = False
    controller._moving = False
    controller._estopped = False
    controller._lock = threading.RLock()
    controller._stop_event = threading.Event()
    controller._follow_thread = None

    result_holder = {}
    worker = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result", controller.move_forward(distance=1.0)
        )
    )
    worker.start()
    deadline = time.monotonic() + 1
    while controller.car.throttle == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    controller.emergency_stop()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert not result_holder["result"].success
    assert "Emergency stop" in result_holder["result"].error
    assert controller.car.throttle == 0
    assert not controller.move_forward(distance=0.1).success
    controller.release_emergency_stop()
