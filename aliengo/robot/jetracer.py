"""Real robot backend for the JetRacer (Jetson Nano RC car + camera).

Implements the same ``RobotController`` Protocol as the mock, so everything
above the seam (LLM, safety, loop, CLI, eval, and LAN server) is reused.  The
car is Ackermann-steered and has no wheel encoders, so turns and distances are
time/steering approximations.  Calibrate every marked value before driving.

The follow loop runs in a background thread doing visual servoing (camera ->
detectNet -> steering/throttle).  Heavy Jetson libraries are imported lazily in
``__init__`` so this module remains importable on the PC development machine.
"""

from __future__ import annotations

import threading
import time

from ..skills.definitions import Posture
from ..vision.detector import Detection
from .interface import RobotState
from .result import SkillResult

# PLACEHOLDER: per-car calibration -- measure/tune these before driving.
THROTTLE_LIMIT = 0.20
DRIVE_THROTTLE = 0.18
SPEED_MPS = 0.5
TURN_STEERING = 0.8
TURN_SECONDS_PER_90DEG = 1.0
STEERING_OFFSET = 0.0

# Follow-loop control tuning.
STEER_KP = 1.6
STOP_HEIGHT_FRAC = 0.6
LOST_FRAMES = 15
MAX_RUNTIME_S = 60.0
LOOP_HZ = 15.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class JetRacerController:
    def __init__(self):
        # Deferred hardware imports (absent on a PC).
        from jetracer.nvidia_racecar import NvidiaRacecar  # noqa: PLC0415

        from ..vision.detector import JetsonDetector  # noqa: PLC0415

        self.car = NvidiaRacecar()
        self.detector = JetsonDetector()
        self._following = False
        self._moving = False
        self._estopped = False
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._follow_thread: threading.Thread | None = None

    # -- motion -----------------------------------------------------------

    def stand_up(self) -> SkillResult:
        with self._lock:
            if self._estopped:
                return self._estop_result("stand_up")
        return SkillResult(
            True, "stand_up", data={"note": "No posture on a car; ready."}
        )

    def sit_down(self) -> SkillResult:
        # Recovery skill: stop motion but intentionally keep an e-stop latched.
        self._halt()
        return SkillResult(
            True, "sit_down", data={"note": "No posture on a car; stopped."}
        )

    def move_forward(self, distance: float) -> SkillResult:
        return self._drive_straight("move_forward", distance, sign=1)

    def move_backward(self, distance: float) -> SkillResult:
        return self._drive_straight("move_backward", distance, sign=-1)

    def turn_left(self, angle: float) -> SkillResult:
        return self._turn("turn_left", angle, steer_sign=-1)

    def turn_right(self, angle: float) -> SkillResult:
        return self._turn("turn_right", angle, steer_sign=1)

    def stop(self) -> SkillResult:
        self._halt()
        return SkillResult(True, "stop")

    def emergency_stop(self) -> None:
        # Latch and halt under the same lock so a concurrent motion start can
        # never re-apply throttle after the stop was activated.
        with self._lock:
            self._estopped = True
            self._halt_locked()

    def release_emergency_stop(self) -> None:
        with self._lock:
            self._estopped = False

    def close(self) -> None:
        self.emergency_stop()
        thread = self._follow_thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self.detector.close()

    # -- perception -------------------------------------------------------

    def find_object(self, label: str) -> SkillResult:
        with self._lock:
            if self._estopped:
                return self._estop_result("find_object")
        match = self._first(label, self.detector.capture_and_detect())
        data = {"found": match is not None, "label": label}
        if match:
            data.update(
                position=match.position, confidence=round(match.confidence, 2)
            )
        return SkillResult(True, "find_object", data=data)

    def detect_objects(self) -> SkillResult:
        with self._lock:
            if self._estopped:
                return self._estop_result("detect_objects")
        objects = [
            {
                "label": detection.label,
                "confidence": round(detection.confidence, 2),
                "position": detection.position,
            }
            for detection in self.detector.capture_and_detect()
        ]
        return SkillResult(True, "detect_objects", data={"objects": objects})

    def follow_person(self) -> SkillResult:
        return self._start_follow("person", "follow_person")

    def follow_object(self, label: str) -> SkillResult:
        return self._start_follow(label, "follow_object")

    def _start_follow(self, label: str, action: str) -> SkillResult:
        if not self._first(label, self.detector.capture_and_detect()):
            return SkillResult(False, action, error=f"No {label} is visible.")
        with self._lock:
            if self._estopped:
                return self._estop_result(action)
            if self._following:
                return SkillResult(
                    False, action, error="Already following; call stop first."
                )
            self._stop_event.clear()
            self._following = True
            self._follow_thread = threading.Thread(
                target=self._follow_loop,
                args=(label,),
                daemon=True,
                name=f"aliengo-follow-{label}",
            )
            self._follow_thread.start()
        return SkillResult(
            True,
            action,
            data={"label": label, "note": "Following; call stop to end."},
        )

    # -- state ------------------------------------------------------------

    def get_state(self) -> RobotState:
        with self._lock:
            return RobotState(
                posture=Posture.STANDING,
                moving=self._moving,
                following=self._following,
                # x/y/heading are unknown without odometry.
                battery_pct=self._read_battery(),
            )

    def reset(self) -> None:
        # Reset never releases the hardware e-stop latch.
        self._halt()

    # -- internals --------------------------------------------------------

    def _read_battery(self) -> float:
        # PLACEHOLDER: JetRacer has no standard battery telemetry. Wire an ADC
        # read here, or return a fixed estimate.
        return 100.0

    @staticmethod
    def _first(label: str, detections: list[Detection]) -> Detection | None:
        matches = [
            detection
            for detection in detections
            if detection.label.lower() == label.lower()
        ]
        return max(matches, key=lambda detection: detection.area) if matches else None

    @staticmethod
    def _estop_result(action: str) -> SkillResult:
        return SkillResult(False, action, error="Emergency stop is active.")

    def _drive_straight(
        self, action: str, distance: float, sign: int
    ) -> SkillResult:
        with self._lock:
            if self._estopped:
                return self._estop_result(action)
            self._stop_event.clear()
            self.car.steering = STEERING_OFFSET
            self.car.throttle = sign * DRIVE_THROTTLE
            self._moving = True

        # Event.wait is interruptible by stop/e-stop, unlike time.sleep.
        interrupted = self._stop_event.wait(distance / SPEED_MPS)
        with self._lock:
            stopped = interrupted or self._stop_event.is_set()
            estopped = self._estopped
            self._halt_locked()
        if stopped:
            reason = (
                "Emergency stop interrupted movement."
                if estopped
                else "Movement was stopped before completion."
            )
            return SkillResult(False, action, error=reason)
        return SkillResult(True, action, data={"distance": distance})

    def _turn(self, action: str, angle: float, steer_sign: int) -> SkillResult:
        with self._lock:
            if self._estopped:
                return self._estop_result(action)
            self._stop_event.clear()
            self.car.steering = _clamp(
                steer_sign * TURN_STEERING + STEERING_OFFSET, -1, 1
            )
            self.car.throttle = DRIVE_THROTTLE
            self._moving = True

        duration = (angle / 90.0) * TURN_SECONDS_PER_90DEG
        interrupted = self._stop_event.wait(duration)
        with self._lock:
            stopped = interrupted or self._stop_event.is_set()
            estopped = self._estopped
            self._halt_locked()
        if stopped:
            reason = (
                "Emergency stop interrupted movement."
                if estopped
                else "Movement was stopped before completion."
            )
            return SkillResult(False, action, error=reason)
        return SkillResult(True, action, data={"angle": angle})

    def _halt(self) -> None:
        with self._lock:
            self._halt_locked()

    def _halt_locked(self) -> None:
        self._stop_event.set()
        self.car.throttle = 0.0
        self.car.steering = STEERING_OFFSET
        self._moving = False
        self._following = False

    def _follow_loop(self, label: str) -> None:
        period = 1.0 / LOOP_HZ
        started = time.monotonic()
        lost = 0
        try:
            while not self._stop_event.is_set():
                if time.monotonic() - started > MAX_RUNTIME_S:
                    break
                target = self._first(label, self.detector.capture_and_detect())
                with self._lock:
                    if self._estopped or self._stop_event.is_set():
                        break
                    if target is None:
                        lost += 1
                        self.car.throttle = 0.0
                        self._moving = False
                    else:
                        lost = 0
                        self.car.steering = _clamp(
                            STEER_KP * (target.cx - 0.5) + STEERING_OFFSET,
                            -1,
                            1,
                        )
                        if target.height >= STOP_HEIGHT_FRAC:
                            self.car.throttle = 0.0
                            self._moving = False
                        else:
                            self.car.throttle = _clamp(
                                DRIVE_THROTTLE, -THROTTLE_LIMIT, THROTTLE_LIMIT
                            )
                            self._moving = True
                if target is None and lost >= LOST_FRAMES:
                    break
                self._stop_event.wait(period)
        finally:
            with self._lock:
                self._halt_locked()
