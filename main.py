import asyncio
import json
import os
import numpy as np
import sounddevice as sd
from datetime import datetime

from config import GEMINI_API_KEY
from db import init_db, create_conversation, save_utterance, end_conversation
from gemini_live import GeminiLiveSession, AudioPlayer, MicStreamer
from recorder import BackgroundRecorder
from processor import process_recording
from notifier import notify_voice, notify_screen
from voice_register import register_voice

import time


async def live_conversation():
    """실시간 음성 대화 + 백그라운드 저장"""
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 GEMINI_API_KEY=your_key 를 추가하세요.")
        return

    await init_db()
    conv_id = await create_conversation(mode="live")

    # Gemini Live 연결
    gemini = GeminiLiveSession()
    player = AudioPlayer()
    mic = MicStreamer()
    recorder = BackgroundRecorder()

    transcript_buffer = []

    def on_audio(data):
        player.enqueue(data)

    def on_text(text):
        transcript_buffer.append({"speaker": "gemini", "text": text})
        print(f"  [Gemini] {text}")

    try:
        await gemini.connect()
        player.start()

        loop = asyncio.get_event_loop()
        mic.start(loop)
        recorder.start()

        # 수신 루프 (백그라운드)
        receive_task = asyncio.create_task(
            gemini.receive_loop(on_audio_callback=on_audio, on_text_callback=on_text)
        )

        print("\n🎙 실시간 대화 시작! (Ctrl+C로 종료)\n")

        # 마이크 → Gemini 전송 루프
        while gemini.is_active:
            try:
                chunk = await asyncio.wait_for(mic.get_chunk(), timeout=1.0)
                await gemini.send_audio(chunk)
            except asyncio.TimeoutError:
                continue

    except KeyboardInterrupt:
        print("\n\n⏹ 대화 종료 중...")
    except Exception as e:
        print(f"\n❌ 오류: {e}")
    finally:
        mic.stop()
        player.stop()
        await gemini.close()

        # 백그라운드 녹음 저장 + 처리
        audio, duration = recorder.stop()
        if len(audio) > 0:
            wav_path = recorder.save_wav(audio)
            print(f"\n📁 녹음 저장: {wav_path}")
            print(f"⏱ 녹음 시간: {duration:.1f}초")
            print("🔄 백그라운드 처리 중 (STT + 화자분리)...")

            results = await process_recording(audio)
            for r in results:
                await save_utterance(conv_id, r["speaker"], r["text"], r["start"], r["end"])
                print(f"  [{r['speaker']}] {r['text']}")

            await end_conversation(conv_id, duration)
            print(f"\n✅ 대화 #{conv_id} 저장 완료 ({len(results)}개 발화)")


async def timed_recording(minutes: int = 5, notify_mode: str = "screen"):
    """타이머 녹음: 지정 시간 녹음 후 처리 + 알림"""
    await init_db()
    conv_id = await create_conversation(mode="timed")

    recorder = BackgroundRecorder()
    recorder.start()

    print(f"\n🎙 {minutes}분 녹음 시작...")
    print(f"   알림 방식: {notify_mode}")

    try:
        await asyncio.sleep(minutes * 60)
    except KeyboardInterrupt:
        print("\n⏹ 녹음 조기 종료")

    audio, duration = recorder.stop()

    if len(audio) == 0:
        print("녹음된 내용이 없습니다.")
        return

    wav_path = recorder.save_wav(audio)
    print(f"\n📁 저장: {wav_path} ({duration:.1f}초)")
    print("🔄 처리 중...")

    results = await process_recording(audio)

    # 저장
    for r in results:
        await save_utterance(conv_id, r["speaker"], r["text"], r["start"], r["end"])

    await end_conversation(conv_id, duration)

    # 결과 포맷팅
    summary_lines = []
    for r in results:
        summary_lines.append(f"[{r['speaker']}] {r['text']}")
    summary = "\n".join(summary_lines)

    print(f"\n=== 대화 내용 ({len(results)}개 발화) ===")
    print(summary)

    # 알림
    if notify_mode in ("voice", "both"):
        short_summary = f"{minutes}분 대화 완료. 총 {len(results)}개 발화가 기록되었습니다."
        await notify_voice(short_summary)

    if notify_mode in ("screen", "both"):
        await notify_screen(summary)

    print(f"\n✅ 대화 #{conv_id} 저장 완료")


async def register_my_voice():
    """내 음성 등록"""
    await init_db()
    await register_voice("나", is_owner=True)


def main():
    import sys

    if len(sys.argv) < 2:
        print("""
사용법:
  python main.py live          실시간 대화
  python main.py record [분]   타이머 녹음 (기본 5분)
  python main.py register      내 음성 등록
  python main.py server        웹 UI 서버 시작
        """)
        return

    cmd = sys.argv[1]

    if cmd == "live":
        asyncio.run(live_conversation())
    elif cmd == "record":
        minutes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        mode = sys.argv[3] if len(sys.argv) > 3 else "screen"
        asyncio.run(timed_recording(minutes, mode))
    elif cmd == "register":
        asyncio.run(register_my_voice())
    elif cmd == "server":
        import uvicorn
        from server import app
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print(f"알 수 없는 명령: {cmd}")


if __name__ == "__main__":
    main()
