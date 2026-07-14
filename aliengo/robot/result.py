from dataclasses import dataclass


@dataclass
class SkillResult:
    success: bool
    action: str
    data: dict | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        out: dict = {"success": self.success, "action": self.action}
        if self.data is not None:
            out["data"] = self.data
        if self.error is not None:
            out["error"] = self.error
        return out
