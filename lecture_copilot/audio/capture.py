from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyaudiowpatch as pyaudio


@dataclass
class LoopbackDevice:
    index: int
    name: str
    channels: int
    sample_rate: int


def list_loopback_devices() -> list[LoopbackDevice]:
    devices: list[LoopbackDevice] = []
    with pyaudio.PyAudio() as p:
        for info in p.get_loopback_device_info_generator():
            devices.append(
                LoopbackDevice(
                    index=int(info["index"]),
                    name=str(info["name"]),
                    channels=int(info["maxInputChannels"]),
                    sample_rate=int(info["defaultSampleRate"]),
                )
            )
    return devices


def default_loopback() -> LoopbackDevice:
    devices = list_loopback_devices()
    if not devices:
        raise RuntimeError(
            "Не знайдено WASAPI loopback-пристрій. Перевір, що в системі є вихід звуку."
        )
    with pyaudio.PyAudio() as p:
        try:
            wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
            default_name = str(default_out["name"])
        except Exception:
            return devices[0]
        for dev in devices:
            if default_name in dev.name:
                return dev
    return devices[0]


def resample_mono(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return audio.astype(np.float32)
    duration = audio.shape[0] / src_rate
    dst_len = max(1, int(round(duration * dst_rate)))
    x_old = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    return np.interp(x_new, x_old, audio.astype(np.float32)).astype(np.float32)


class LoopbackCapture:
    """WASAPI loopback: системний звук (Webex/Teams/YouTube), включно з Bluetooth-виходом."""

    def __init__(self, device: LoopbackDevice | None = None, frames_per_buffer: int = 2048):
        self.device = device or default_loopback()
        self.frames_per_buffer = frames_per_buffer
        self._pa: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None

    def start(self) -> None:
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.device.channels,
            rate=self.device.sample_rate,
            input=True,
            input_device_index=self.device.index,
            frames_per_buffer=self.frames_per_buffer,
        )

    def read(self) -> np.ndarray:
        if self._stream is None:
            raise RuntimeError("Capture is not started")
        raw = self._stream.read(self.frames_per_buffer, exception_on_overflow=False)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if self.device.channels > 1:
            samples = samples.reshape(-1, self.device.channels).mean(axis=1)
        return samples

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
