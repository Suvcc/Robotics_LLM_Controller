"""Camera object detector for the JetRacer (Jetson only).

Wraps a jetson-inference `detectNet` and a `jetcam` camera behind a small,
resolution-independent `Detection` type. All heavy imports are deferred into
`JetsonDetector.__init__`, so importing this module on a PC without the Jetson
libraries is safe (tests and the mock CLI never touch the hardware path).

Coordinates are normalized to 0..1 (cx, cy, width, height) so the follow-loop
control gains don't depend on camera resolution.
"""
from dataclasses import dataclass

# ── PLACEHOLDER: camera / model settings — set these for your hardware ──
CAMERA_SENSOR_ID = 0        # CSI sensor id (or USB device index)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
DETECTNET_MODEL = "ssd-mobilenet-v2"   # pretrained on COCO (person, bottle, …)
DETECTNET_THRESHOLD = 0.5


@dataclass
class Detection:
    label: str
    confidence: float
    cx: float      # box center x, 0 (left) .. 1 (right)
    cy: float      # box center y, 0 (top) .. 1 (bottom)
    width: float   # 0 .. 1
    height: float  # 0 .. 1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def position(self) -> str:
        """Coarse horizontal position, matching the mock's find_object output."""
        if self.cx < 0.4:
            return "left"
        if self.cx > 0.6:
            return "right"
        return "center"


class JetsonDetector:
    """Real detector. Only instantiate on the Jetson (needs GPU + camera)."""

    def __init__(
        self,
        model: str = DETECTNET_MODEL,
        threshold: float = DETECTNET_THRESHOLD,
        sensor_id: int = CAMERA_SENSOR_ID,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS,
    ):
        # Deferred imports: absent on a normal PC, present on the Jetson.
        from jetcam.csi_camera import CSICamera  # noqa: PLC0415
        from jetson_inference import detectNet  # noqa: PLC0415

        self._net = detectNet(model, threshold=threshold)
        self._camera = CSICamera(
            width=width, height=height, capture_fps=fps, capture_device=sensor_id
        )
        self._w = width
        self._h = height

    def capture_and_detect(self) -> list[Detection]:
        """Grab one frame and return all detections (normalized coords)."""
        from jetson_utils import cudaFromNumpy  # noqa: PLC0415

        frame = self._camera.read()  # numpy BGR
        cuda_img = cudaFromNumpy(frame)
        detections = []
        for d in self._net.Detect(cuda_img, overlay="none"):
            detections.append(
                Detection(
                    label=self._net.GetClassDesc(d.ClassID),
                    confidence=float(d.Confidence),
                    cx=d.Center[0] / self._w,
                    cy=d.Center[1] / self._h,
                    width=d.Width / self._w,
                    height=d.Height / self._h,
                )
            )
        return detections

    def close(self) -> None:
        cam = getattr(self, "_camera", None)
        if cam is not None:
            try:
                cam.running = False
            except Exception:
                pass
