import asyncio
import json
import base64
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from config import GEMINI_API_KEY, SAMPLE_RATE_INPUT
from db import (
    init_db, create_conversation, save_utterance,
    end_conversation, get_conversation, get_recent_conversations, get_voice_profiles,
)
from gemini_live import GeminiLiveSession
from recorder import BackgroundRecorder
from processor import process_recording
from voice_register import quick_register
from notifier import generate_tts_bytes

app = FastAPI(title="음성 비서")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).parent / "web_ui"


@app.on_event("startup")
async def startup():
    await init_db()


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/conversations")
async def list_conversations():
    return await get_recent_conversations()


@app.get("/api/conversations/{conv_id}")
async def get_conv(conv_id: int):
    return await get_conversation(conv_id)


@app.get("/api/profiles")
async def list_profiles():
    return await get_voice_profiles()


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    """웹소켓: 실시간 Gemini 음성 대화"""
    await ws.accept()

    if not GEMINI_API_KEY:
        await ws.send_json({"type": "error", "message": "GEMINI_API_KEY not set"})
        await ws.close()
        return

    conv_id = await create_conversation(mode="live")
    gemini = GeminiLiveSession()
    audio_buffer = []
    start_time = None

    async def on_audio(data):
        encoded = base64.b64encode(data).decode()
        await ws.send_json({"type": "audio", "data": encoded})

    async def on_text(text):
        await ws.send_json({"type": "text", "speaker": "gemini", "text": text})

    try:
        await gemini.connect()
        await ws.send_json({"type": "connected"})
        start_time = asyncio.get_event_loop().time()

        # 수신 루프
        receive_task = asyncio.create_task(
            gemini.receive_loop(
                on_audio_callback=lambda d: asyncio.ensure_future(on_audio(d)),
                on_text_callback=lambda t: asyncio.ensure_future(on_text(t)),
            )
        )

        # 클라이언트에서 오디오 수신
        while True:
            msg = await ws.receive_json()

            if msg["type"] == "audio":
                pcm_data = base64.b64decode(msg["data"])
                await gemini.send_audio(pcm_data)
                # 백그라운드 버퍼에 저장
                audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer.append(audio_array)

            elif msg["type"] == "stop":
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        await gemini.close()
        duration = asyncio.get_event_loop().time() - start_time if start_time else 0

        # 백그라운드 처리
        if audio_buffer:
            full_audio = np.concatenate(audio_buffer)
            results = await process_recording(full_audio)
            for r in results:
                await save_utterance(conv_id, r["speaker"], r["text"], r["start"], r["end"])
            await end_conversation(conv_id, duration)

        try:
            await ws.send_json({
                "type": "session_end",
                "conversation_id": conv_id,
                "duration": round(duration, 1),
            })
        except:
            pass


@app.websocket("/ws/record")
async def ws_record(ws: WebSocket):
    """웹소켓: 타이머 녹음"""
    await ws.accept()
    audio_buffer = []

    try:
        while True:
            msg = await ws.receive_json()

            if msg["type"] == "audio":
                pcm_data = base64.b64decode(msg["data"])
                audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer.append(audio_array)

            elif msg["type"] == "stop":
                break

    except WebSocketDisconnect:
        pass

    if audio_buffer:
        full_audio = np.concatenate(audio_buffer)
        conv_id = await create_conversation(mode="timed")
        results = await process_recording(full_audio)
        duration = len(full_audio) / SAMPLE_RATE_INPUT

        for r in results:
            await save_utterance(conv_id, r["speaker"], r["text"], r["start"], r["end"])
        await end_conversation(conv_id, duration)

        # 결과 전송
        notify_mode = "screen"
        try:
            await ws.send_json({
                "type": "result",
                "conversation_id": conv_id,
                "duration": round(duration, 1),
                "utterances": results,
            })
        except:
            pass


@app.websocket("/ws/register")
async def ws_register(ws: WebSocket):
    """웹소켓: 음성 등록"""
    await ws.accept()

    try:
        msg = await ws.receive_json()
        name = msg.get("name", "나")
        is_owner = msg.get("is_owner", False)
        audio_b64 = msg.get("audio", "")

        pcm_data = base64.b64decode(audio_b64)
        audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

        path = await quick_register(name, audio_array, is_owner=is_owner)
        await ws.send_json({"type": "registered", "name": name, "path": path})

    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})
    finally:
        await ws.close()


# Static files
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
