import asyncio
import numpy as np
import sounddevice as sd
import threading
import queue
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_MODEL, SAMPLE_RATE_INPUT, SAMPLE_RATE_OUTPUT


class GeminiLiveSession:
    """Gemini Live API를 통한 실시간 음성 대화"""

    def __init__(self, system_instruction: str = None):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.session = None
        self.is_active = False
        self._playback_queue = queue.Queue()

        if system_instruction is None:
            system_instruction = (
                "당신은 한국어 음성 비서입니다. "
                "자연스럽고 간결하게 한국어로 대답하세요. "
                "존댓말을 사용하세요."
            )
        self.system_instruction = system_instruction

    async def connect(self):
        """Gemini Live 세션 연결"""
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Kore"  # 한국어 음성
                    )
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=self.system_instruction)]
            ),
        )

        self.session = await self.client.aio.live.connect(
            model=GEMINI_MODEL,
            config=config
        )
        self.is_active = True
        print("[Gemini] 연결됨")

    async def send_audio(self, audio_chunk: bytes):
        """PCM 오디오 청크 전송"""
        if self.session and self.is_active:
            await self.session.send(
                input=types.LiveClientRealtimeInput(
                    media_chunks=[
                        types.Blob(data=audio_chunk, mime_type="audio/pcm;rate=16000")
                    ]
                )
            )

    async def receive_loop(self, on_audio_callback=None, on_text_callback=None):
        """응답 수신 루프"""
        if not self.session:
            return

        try:
            async for response in self.session.receive():
                if not self.is_active:
                    break

                server_content = response.server_content
                if server_content is None:
                    continue

                if server_content.model_turn:
                    for part in server_content.model_turn.parts:
                        if part.inline_data:
                            # 오디오 응답
                            audio_data = part.inline_data.data
                            if on_audio_callback:
                                on_audio_callback(audio_data)
                            else:
                                self._playback_queue.put(audio_data)
                        elif part.text:
                            if on_text_callback:
                                on_text_callback(part.text)

                if server_content.turn_complete:
                    # 턴 완료 마커
                    self._playback_queue.put(None)

        except Exception as e:
            if self.is_active:
                print(f"[Gemini] 수신 오류: {e}")

    async def close(self):
        """세션 종료"""
        self.is_active = False
        if self.session:
            await self.session.close()
            print("[Gemini] 연결 종료")


class AudioPlayer:
    """Gemini 오디오 응답 재생"""

    def __init__(self):
        self._stream = None
        self._queue = queue.Queue()
        self._playing = False

    def start(self):
        self._playing = True
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE_OUTPUT,
            channels=1,
            dtype="int16",
            blocksize=4800,
        )
        self._stream.start()
        threading.Thread(target=self._play_loop, daemon=True).start()

    def _play_loop(self):
        while self._playing:
            try:
                data = self._queue.get(timeout=0.5)
                if data is None:
                    continue
                audio_array = np.frombuffer(data, dtype=np.int16)
                self._stream.write(audio_array.reshape(-1, 1))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Player] 재생 오류: {e}")

    def enqueue(self, audio_data: bytes):
        self._queue.put(audio_data)

    def stop(self):
        self._playing = False
        if self._stream:
            self._stream.stop()
            self._stream.close()


class MicStreamer:
    """마이크 입력을 Gemini로 스트리밍"""

    def __init__(self):
        self._stream = None
        self._audio_queue = asyncio.Queue()
        self.is_streaming = False

    def start(self, loop: asyncio.AbstractEventLoop):
        self.is_streaming = True
        self._loop = loop

        def callback(indata, frames, time_info, status):
            if self.is_streaming:
                pcm_bytes = (indata * 32767).astype(np.int16).tobytes()
                self._loop.call_soon_threadsafe(self._audio_queue.put_nowait, pcm_bytes)

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE_INPUT,
            channels=1,
            dtype="float32",
            callback=callback,
            blocksize=int(SAMPLE_RATE_INPUT * 0.5)
        )
        self._stream.start()

    async def get_chunk(self) -> bytes:
        return await self._audio_queue.get()

    def stop(self):
        self.is_streaming = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
