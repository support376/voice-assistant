import asyncio
import edge_tts
import sounddevice as sd
import numpy as np
import io
import wave
import tempfile
import os


async def notify_voice(text: str, voice: str = "ko-KR-SunHiNeural"):
    """Edge TTS로 텍스트를 음성으로 알림 (끊김 없이 전체 생성 후 재생)"""
    # 1. 전체 음성 파일 먼저 생성
    communicate = edge_tts.Communicate(text, voice)
    tmp_path = os.path.join(tempfile.gettempdir(), "notify_tts.mp3")
    await communicate.save(tmp_path)

    # 2. ffmpeg 없이 재생하기 위해 edge-tts의 raw 스트림 사용
    audio_chunks = []
    communicate = edge_tts.Communicate(text, voice, receive_timeout=30)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])

    if not audio_chunks:
        print("[알림] 음성 생성 실패")
        return

    # MP3 → 임시 파일 저장 후 외부 재생
    audio_bytes = b"".join(audio_chunks)
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    # 시스템 기본 플레이어로 재생
    if os.name == 'nt':
        os.system(f'start /min "" "wmplayer" "{tmp_path}"')
    else:
        os.system(f'ffplay -nodisp -autoexit "{tmp_path}" 2>/dev/null &')

    print(f"[알림] 음성 재생: {text[:30]}...")


async def notify_screen(text: str) -> dict:
    """화면 알림용 데이터 반환 (웹 UI에서 표시)"""
    return {
        "type": "notification",
        "text": text,
    }


async def generate_tts_bytes(text: str, voice: str = "ko-KR-SunHiNeural") -> bytes:
    """TTS 음성을 bytes로 반환 (웹소켓 전송용)"""
    audio_chunks = []
    communicate = edge_tts.Communicate(text, voice, receive_timeout=30)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
    return b"".join(audio_chunks)
