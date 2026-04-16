import numpy as np
import asyncio
from faster_whisper import WhisperModel
from config import SAMPLE_RATE_INPUT
from speaker import extract_embedding, identify_speaker

_whisper_model = None


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        _whisper_model = WhisperModel("large-v3", device=device, compute_type=compute)
    return _whisper_model


async def process_recording(audio: np.ndarray, sr: int = SAMPLE_RATE_INPUT) -> list:
    """
    녹음된 오디오를 처리: STT + 화자분리 + 화자식별
    반환: [{"speaker": "나", "text": "...", "start": 0.0, "end": 2.5}, ...]
    """
    if len(audio) < sr:  # 1초 미만이면 무시
        return []

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _process_sync, audio, sr)
    return result


def _process_sync(audio: np.ndarray, sr: int) -> list:
    """동기 처리 (스레드에서 실행)"""
    model = get_whisper()

    # 1. Whisper STT
    segments, info = model.transcribe(
        audio,
        language="ko",
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
    )

    results = []
    for segment in segments:
        # 2. 세그먼트별 오디오 추출
        start_sample = int(segment.start * sr)
        end_sample = int(segment.end * sr)
        segment_audio = audio[start_sample:end_sample]

        # 3. 화자 식별 (세그먼트가 1초 이상일 때만)
        speaker = "unknown"
        if len(segment_audio) >= sr:
            try:
                emb = extract_embedding(segment_audio, sr)
                speaker = identify_speaker(emb)
            except Exception:
                speaker = "unknown"

        results.append({
            "speaker": speaker,
            "text": segment.text.strip(),
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
        })

    return results


def transcribe_simple(audio: np.ndarray, sr: int = SAMPLE_RATE_INPUT) -> str:
    """단순 STT (화자분리 없이 텍스트만)"""
    model = get_whisper()
    segments, _ = model.transcribe(audio, language="ko", vad_filter=True)
    return " ".join(s.text.strip() for s in segments)
