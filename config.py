import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# Data files
DB_PATH = os.path.join(DATA_DIR, "library.db")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "vectors.index")
FAISS_ID_MAP_PATH = os.path.join(DATA_DIR, "id_map.json")  # maps FAISS int ID -> DB song ID

# CLAP model
CLAP_MODEL_ID = "laion/clap-htsat-unfused"
EMBEDDING_DIM = 512

# Audio processing
SAMPLE_RATE = 48000          # CLAP expects 48kHz
MAX_DURATION_SECONDS = 300   # cap at 5 minutes for long files
MIN_DURATION_SECONDS = 5     # reject clips shorter than this
MIN_RMS_ENERGY = 1e-4        # reject near-silent audio

# Supported upload formats
SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
# Formats that need ffmpeg conversion before librosa can load reliably
NEEDS_CONVERSION = {".wma", ".aac", ".m4a"}

# Flask
MAX_UPLOAD_BYTES = 50 * 1024 * 1024   # 50MB
FLASK_PORT = 5000

# Search
DEFAULT_TOP_N = 10
LOW_CONFIDENCE_THRESHOLD = 0.60        # warn user if best match is below this

# Ensure dirs exist at import time
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
