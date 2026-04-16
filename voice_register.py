import numpy as np
import sounddevice as sd
import asyncio
from config import SAMPLE_RATE_INPUT
from speaker import extract_embedding, save_profile
from db import save_voice_profile


REGISTER_PHRASES = [
    "안녕하세요, 오늘 날씨가 좋네요.",
    "내일 회의는 오후 세 시에 시작합니다.",
    "프로젝트 진행 상황을 공유하겠습니다.",
]


def record_audio(duration_sec: float = 5.0) -> np.ndarray:
    """마이크에서 오디오 녹음"""
    print(f"  녹음 중... ({duration_sec}초)")
    audio = sd.rec(
        int(duration_sec * SAMPLE_RATE_INPUT),
        samplerate=SAMPLE_RATE_INPUT,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    print("  녹음 완료")
    return audio.flatten()


async def register_voice(name: str, is_owner: bool = False) -> str:
    """음성 등록: 여러 문장 녹음 → 평균 임베딩 저장"""
    embeddings = []

    print(f"\n=== 음성 등록: {name} ===")
    print(f"아래 문장들을 하나씩 읽어주세요.\n")

    for i, phrase in enumerate(REGISTER_PHRASES):
        print(f"[{i+1}/{len(REGISTER_PHRASES)}] \"{phrase}\"")
        input("  준비되면 Enter를 누르세요...")
        audio = record_audio(duration_sec=5.0)

        if np.abs(audio).max() < 0.01:
            print("  ⚠ 소리가 감지되지 않았습니다. 다시 시도하세요.")
            audio = record_audio(duration_sec=5.0)

        emb = extract_embedding(audio)
        embeddings.append(emb)
        print(f"  ✓ 임베딩 추출 완료\n")

    avg_embedding = np.mean(embeddings, axis=0)
    path = save_profile(name, avg_embedding)
    await save_voice_profile(name, path, is_owner=is_owner)

    print(f"=== '{name}' 음성 등록 완료 ===\n")
    return path


async def quick_register(name: str, audio_array: np.ndarray, is_owner: bool = False) -> str:
    """오디오 배열에서 바로 등록 (웹 UI용)"""
    emb = extract_embedding(audio_array)
    path = save_profile(name, emb)
    await save_voice_profile(name, path, is_owner=is_owner)
    return path
