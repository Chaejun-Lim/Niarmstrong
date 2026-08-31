import serial
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path


LEADER_BY_ID_PORT = "/dev/serial/by-id/usb-Arduino_RaspberryPi_Pico_472803531C5F2641-if00"


class LeaderArmInterface(ABC):
    """Base interface for a leader arm reader."""

    def __init__(self):
        self.last_valid_action: np.ndarray | None = None

    @abstractmethod
    def read_action(self) -> np.ndarray:
        """Read the latest leader action."""

    def close(self):
        pass


class SerialLeaderArmInterface(LeaderArmInterface):
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.0):
        super().__init__()
        resolved_port = self._resolve_leader_port(port)
        try:
            self.ser = serial.Serial(resolved_port, baudrate, timeout=timeout, dsrdtr=False)
            print(f"[HW] 리더암 시리얼 연결 성공 ({resolved_port})")
        except serial.SerialException as e:
            print(f"[에러] 리더암 시리얼 포트를 열 수 없습니다: {e}")
            self.ser = None
        self._rx_buffer = bytearray()

    @staticmethod
    def _resolve_leader_port(port: str) -> str:
        """Prefer a stable by-id path for the leader device when available."""
        preferred = Path(LEADER_BY_ID_PORT)
        if preferred.exists() and port.startswith("/dev/ttyACM"):
            return str(preferred)
        return port

    @property
    def is_connected(self) -> bool:
        return self.ser is not None and self.ser.is_open

    def read_action_sample(self) -> tuple[np.ndarray, bool]:
        """Return the newest complete 7-value sample and whether it arrived this call.

        Non-blocking reads keep a missing serial message from changing the dataset's
        declared FPS. Callers must discard non-fresh samples.
        """
        if not self.is_connected:
            raise RuntimeError("Leader arm is not connected.")

        try:
            available = self.ser.in_waiting
            if available:
                self._rx_buffer.extend(self.ser.read(available))

            newline_idx = self._rx_buffer.rfind(b"\n")
            raw_line = None
            if newline_idx != -1:
                complete = bytes(self._rx_buffer[: newline_idx + 1])
                del self._rx_buffer[: newline_idx + 1]
                lines = [line.strip() for line in complete.splitlines() if line.strip()]
                if lines:
                    raw_line = lines[-1]
            elif len(self._rx_buffer) > 4096:
                # Guard against unbounded growth when malformed data arrives.
                self._rx_buffer = self._rx_buffer[-1024:]

            if raw_line is None:
                if self.last_valid_action is None:
                    raise RuntimeError("Leader arm has not produced any action samples yet.")
                return self.last_valid_action.copy(), False

            values = np.fromstring(raw_line.decode("utf-8", errors="strict"), sep=",", dtype=np.float32)
            if values.shape != (7,) or not np.isfinite(values).all():
                if self.last_valid_action is None:
                    raise RuntimeError("Leader arm produced an invalid action sample before any valid data was received.")
                return self.last_valid_action.copy(), False
            self.last_valid_action = values
            return values.copy(), True
        except (UnicodeDecodeError, ValueError, serial.SerialException):
            if self.last_valid_action is None:
                raise RuntimeError("Leader arm serial read failed before any action sample was received.")
            return self.last_valid_action.copy(), False

    def read_action(self) -> np.ndarray:
        return self.read_action_sample()[0]

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[HW] 리더암 시리얼 연결 종료")
