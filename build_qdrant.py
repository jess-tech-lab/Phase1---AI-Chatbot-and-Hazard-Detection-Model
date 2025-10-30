import os
from simple_llm.agents.delegator import DelegatorClient
# TODO: not implemented yet in the simple-llm package
# from simple_llm.embeddings.google import gemini_embedding
from simple_llm.embeddings.openai import openai_embedding
import dotenv

from qdrant_client import QdrantClient, models

# TODO: Rename keys.env to .env and put the OPENAI_API_KEY inside together with the GEMINI_API_KEY
dotenv.load_dotenv(".env")

COLLECTION_NAME = "persona-delegation"
QDRANT_PATH = os.path.abspath("prompt_db_qdrant")

# === Global cached client (singleton pattern) ===
_qdrant_client_instance = None

def get_qdrant_client(recreate: bool = False) -> QdrantClient:
    global _qdrant_client_instance
    if _qdrant_client_instance is None:
        _qdrant_client_instance = QdrantClient(path=QDRANT_PATH)

    client = _qdrant_client_instance

    try:
        print(f"Checking for existing collection '{COLLECTION_NAME}'...")

        # Check if the collection exists. If it does, delete it.
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            print("Creating new collection...")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
            )
            print("Collection created successfully.")
        else:
            if recreate:
                print("Collection found. Deleting old collection...")
                client.delete_collection(collection_name=COLLECTION_NAME)

                print("Recreating new collection...")
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
                )
                print("Collection recreated successfully.")

        return client
    except Exception as e:
        # Catch any general errors during this process
        print(f"Collection initialization error: {e}")

def build_persona_collection(db: DelegatorClient):
    """
    Builds and indexes the persona collection.
    """
    print(f"\nIndexing prompts into collection '{COLLECTION_NAME}'...")

    db.build_collection(
        prompts=[
            "diabetes",
            "high blood pressure",
            "high cholesterol",
            "86 years old",
            "Stage 2 lung cancer",
            "81 years old",
            "Kidney stones",
            "epilepsy",
        ],
        categories=["1", "1", "1", "1", "2", "2", "2", "2"],
    )

    print("Indexing complete. The vector database is ready for delegation.")

def get_delegator_client(recreate: bool = False, rebuild: bool = False) -> DelegatorClient:
    """
    Returns a DelegatorClient (wrapper for embeddings and retrieval).
    If recreate=True, rebuilds the collection first.
    If rebuild=True, re-indexes the collection prompts.
    """
    client = get_qdrant_client(recreate=recreate)
    db = DelegatorClient(client, COLLECTION_NAME, openai_embedding)

    if rebuild:
        build_persona_collection(db)
    return db
