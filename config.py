import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# Audio settings
SAMPLE_RATE_INPUT = 16000    # 마이크 입력
SAMPLE_RATE_OUTPUT = 24000   # Gemini 출력
CHANNELS = 1
CHUNK_DURATION = 0.5         # 초 단위 청크

# Speaker recognition
SPEAKER_THRESHOLD = 0.75     # 코사인 유사도 임계값
EMBEDDING_DIM = 192          # ECAPA-TDNN

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VOICE_PROFILES_DIR = os.path.join(DATA_DIR, "voice_profiles")
RECORDINGS_DIR = os.path.join(DATA_DIR, "recordings")
DB_PATH = os.path.join(DATA_DIR, "conversations.db")

os.makedirs(VOICE_PROFILES_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)
