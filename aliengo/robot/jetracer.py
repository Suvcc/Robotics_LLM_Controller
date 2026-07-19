"""Real robot backend for the JetRacer (Jetson Nano RC car + camera).

Implements the same `RobotController` Protocol as the mock, so everything above
the seam (LLM, safety, loop, CLI, eval) is reused unchanged. The car is
Ackermann-steered: it cannot rotate in place and has no wheel encoders, so
turns and distances are time/steering approximations — see the PLACEHOLDERs.

The follow loop runs in a background thread doing visual servoing (camera →
detectNet → steering/throttle); the LLM only decides when to start/stop it.

Heavy hardware libraries (`jetracer`, `jetson_inference`, `jetcam`) are imported
lazily in __init__, so importing THIS module on a PC is safe. Only constructing
`JetRacerController` requires the Jetson.
"""
import threading
import time

from ..skills.definitions import Posture
from ..vision.detector import Detection
from .interface import RobotState
from .result import SkillResult

# ── PLACEHOLDER: per-car calibration — measure/tune these before driving ──
THROTTLE_LIMIT = 0.20     # hard safety cap on |throttle|; start low
DRIVE_THROTTLE = 0.18     # nominal forward throttle for move/turn
SPEED_MPS = 0.5           # measured car speed (m/s) at DRIVE_THROTTLE, for distance→time
TURN_STEERING = 0.8       # steering magnitude used to approximate a turn
TURN_SECONDS_PER_90DEG = 1.0   # seconds of steer-drive that yields ~90° of heading change
STEERING_OFFSET = 0.0     # trim so steering=0 drives straight
# Follow-loop control tuning:
STEER_KP = 1.6            # steering per unit of horizontal offset from center
STOP_HEIGHT_FRAC = 0.6    # target box height (0..1) at which we're "close enough" → stop advancing
LOST_FRAMES = 15          # consecutive frames without the target before auto-stop
MAX_RUNTIME_S = 60.0      # hard cap on a single follow session
LOOP_HZ = 15.0            # control loop rate


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
        self._stop_event = threading.Event()
        self._follow_thread: threading.Thread | None = None

    # -- motion --------------------------------------------------------------

    def stand_up(self) -> SkillResult:
        # A car has no posture; treat as a satisfied no-op so posture
        # preconditions on movement skills pass.
        return SkillResult(True, "stand_up", data={"note": "No posture on a car; ready."})

    def sit_down(self) -> SkillResult:
        self.emergency_stop()
        return SkillResult(True, "sit_down", data={"note": "No posture on a car; stopped."})

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
        self._halt()

    # -- perception ----------------------------------------------------------

    def find_object(self, label: str) -> SkillResult:
        match = self._first(label, self.detector.capture_and_detect())
        data = {"found": match is not None, "label": label}
        if match:
            data.update(position=match.position, confidence=round(match.confidence, 2))
        return SkillResult(True, "find_object", data=data)

    def detect_objects(self) -> SkillResult:
        objects = [
            {"label": d.label, "confidence": round(d.confidence, 2), "position": d.position}
            for d in self.detector.capture_and_detect()
        ]
        return SkillResult(True, "detect_objects", data={"objects": objects})

    def follow_person(self) -> SkillResult:
        return self.follow_object("person")

    def follow_object(self, label: str) -> SkillResult:
        if not self._first(label, self.detector.capture_and_detect()):
            return SkillResult(False, "follow_object", error=f"No {label} is visible.")
        self._stop_event.clear()
        self._following = True
        self._follow_thread = threading.Thread(
            target=self._follow_loop, args=(label,), daemon=True
        )
        self._follow_thread.start()
        return SkillResult(True, "follow_object", data={"label": label, "note": "Following; call stop to end."})

    # -- state ---------------------------------------------------------------

    def get_state(self) -> RobotState:
        return RobotState(
            posture=Posture.STANDING,   # cars are always "ready"
            moving=self._moving,
            following=self._following,
            # x/y/heading unknown without odometry — left at 0.
            battery_pct=self._read_battery(),
        )

    def reset(self) -> None:
        self._halt()

    # -- internals -----------------------------------------------------------

    def _read_battery(self) -> float:
        # PLACEHOLDER: JetRacer has no standard battery telemetry. Wire your ADC
        # read here, or return a fixed estimate.
        return 100.0

    @staticmethod
    def _first(label: str, detections: list[Detection]) -> Detection | None:
        matches = [d for d in detections if d.label.lower() == label.lower()]
        return max(matches, key=lambda d: d.area) if matches else None

    def _drive_straight(self, action: str, distance: float, sign: int) -> SkillResult:
        self.car.steering = STEERING_OFFSET
        self.car.throttle = sign * DRIVE_THROTTLE
        self._moving = True
        time.sleep(distance / SPEED_MPS)   # PLACEHOLDER kinematics (no encoders)
        self._halt()
        return SkillResult(True, action, data={"distance": distance})

    def _turn(self, action: str, angle: float, steer_sign: int) -> SkillResult:
        self.car.steering = _clamp(steer_sign * TURN_STEERING + STEERING_OFFSET, -1, 1)
        self.car.throttle = DRIVE_THROTTLE
        self._moving = True
        time.sleep((angle / 90.0) * TURN_SECONDS_PER_90DEG)  # PLACEHOLDER approximation
        self._halt()
        return SkillResult(True, action, data={"angle": angle})

    def _halt(self) -> None:
        self._stop_event.set()   # signal the follow loop to end
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
                if target is None:
                    lost += 1
                    if lost >= LOST_FRAMES:
                        break
                    self.car.throttle = 0.0
                    time.sleep(period)
                    continue
                lost = 0
                # Steer toward the target's horizontal center.
                self.car.steering = _clamp(
                    STEER_KP * (target.cx - 0.5) + STEERING_OFFSET, -1, 1
                )
                # Advance while the target is far (small box); ease off as it fills.
                if target.height >= STOP_HEIGHT_FRAC:
                    self.car.throttle = 0.0
                else:
                    self.car.throttle = _clamp(DRIVE_THROTTLE, -THROTTLE_LIMIT, THROTTLE_LIMIT)
                time.sleep(period)
        finally:
            self.car.throttle = 0.0
            self.car.steering = STEERING_OFFSET
            self._moving = False
            self._following = False
