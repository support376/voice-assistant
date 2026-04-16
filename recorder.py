import numpy as np
import sounddevice as sd
import asyncio
import threading
import time
import wave
import os
from datetime import datetime
from config import SAMPLE_RATE_INPUT, CHANNELS, RECORDINGS_DIR


class BackgroundRecorder:
    """백그라운드 녹음기: 실시간 대화와 별도로 전체 오디오 저장"""

    def __init__(self):
        self.is_recording = False
        self.audio_chunks = []
        self.start_time = None
        self._stream = None
        self._thread = None

    def start(self):
        self.is_recording = True
        self.audio_chunks = []
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def _record_loop(self):
        def callback(indata, frames, time_info, status):
            if self.is_recording:
                self.audio_chunks.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE_INPUT,
            channels=CHANNELS,
            dtype="float32",
            callback=callback,
            blocksize=int(SAMPLE_RATE_INPUT * 0.5)
        )
        self._stream.start()
        while self.is_recording:
            time.sleep(0.1)
        self._stream.stop()
        self._stream.close()

    def stop(self) -> tuple:
        """녹음 중지. (오디오 배열, 녹음시간) 반환"""
        self.is_recording = False
        if self._thread:
            self._thread.join(timeout=2)
        duration = time.time() - self.start_time if self.start_time else 0
        if self.audio_chunks:
            audio = np.concatenate(self.audio_chunks, axis=0).flatten()
        else:
            audio = np.array([], dtype=np.float32)
        return audio, duration

    def save_wav(self, audio: np.ndarray, filename: str = None) -> str:
        """WAV 파일로 저장"""
        if filename is None:
            filename = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        path = os.path.join(RECORDINGS_DIR, filename)
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(path, 'w') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE_INPUT)
            wf.writeframes(audio_int16.tobytes())
        return path
