import json
import pickle
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.models import Chunk
from src.core.embeddings import OpenAIEmbedder
from src.retrieval.vector_store import VectorStore

load_dotenv()

CHUNKS_PATH = Path("data/chunks/chunks.json")
INDEX_DIR = Path("data/indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def main():
    raw = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    chunks = [Chunk(**c) for c in raw]

    embedder = OpenAIEmbedder()
    store = VectorStore(embedder=embedder, dim=3072)

    store.add_chunks(chunks)

    with open(INDEX_DIR / "store.pkl", "wb") as f:
        pickle.dump(store, f)

    print(f"Built FAISS index with {len(chunks)} chunks -> {INDEX_DIR / 'store.pkl'}")

if __name__ == "__main__":
    main()
