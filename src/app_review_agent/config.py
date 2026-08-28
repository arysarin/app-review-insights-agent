"""Central configuration, loaded once from environment variables.

Every other module should import settings from here rather than
calling os.getenv directly — keeps config in one place as the
project grows past Step 1.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

TARGET_APP_ID = os.getenv("TARGET_APP_ID", "com.example.app")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "app_reviews")

REVIEWS_CSV_PATH = DATA_DIR / "reviews.csv"
CHECKPOINT_DB_PATH = DATA_DIR / "checkpoints.sqlite"

CATEGORIES = ["crash", "ux", "billing", "praise", "feature_request"]
