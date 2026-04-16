import numpy as np
import os
import torch
from config import VOICE_PROFILES_DIR, SPEAKER_THRESHOLD, SAMPLE_RATE_INPUT

_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        from speechbrain.inference.speaker import EncoderClassifier
        _classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(VOICE_PROFILES_DIR, ".model_cache"),
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
    return _classifier


def extract_embedding(audio_array: np.ndarray, sr: int = SAMPLE_RATE_INPUT) -> np.ndarray:
    """오디오에서 화자 임베딩 추출 (192차원)"""
    classifier = get_classifier()
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    if audio_array.max() > 1.0:
        audio_array = audio_array / 32768.0
    tensor = torch.tensor(audio_array).unsqueeze(0)
    embedding = classifier.encode_batch(tensor)
    return embedding.squeeze().cpu().numpy()


def save_profile(name: str, embedding: np.ndarray) -> str:
    """음성 프로필 저장"""
    path = os.path.join(VOICE_PROFILES_DIR, f"{name}.npy")
    np.save(path, embedding)
    return path


def load_all_profiles() -> dict:
    """저장된 모든 프로필 로드"""
    profiles = {}
    for f in os.listdir(VOICE_PROFILES_DIR):
        if f.endswith(".npy"):
            name = f[:-4]
            profiles[name] = np.load(os.path.join(VOICE_PROFILES_DIR, f))
    return profiles


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def identify_speaker(embedding: np.ndarray) -> str:
    """임베딩으로 화자 식별. 매칭 안 되면 'unknown' 반환"""
    profiles = load_all_profiles()
    if not profiles:
        return "unknown"

    best_name = "unknown"
    best_score = 0.0

    for name, profile_emb in profiles.items():
        score = cosine_similarity(embedding, profile_emb)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= SPEAKER_THRESHOLD:
        return best_name
    return "unknown"
