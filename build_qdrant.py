from simple_llm.agents.delegator import DelegatorClient
# from simple_llm.embeddings.google import gemini_embedding (not implemented yet)
from simple_llm.embeddings.openai import openai_embedding
import dotenv

from qdrant_client import QdrantClient, models

# Rename keys.env to .env and put the OPENAI_API_KEY inside together with the GEMINI_API_KEY
dotenv.load_dotenv(".env")

COLLECTION_NAME = "persona-delegation"
client = QdrantClient(path="prompt_db-qdrant")

# --- Replacement for client.recreate_collection() ---
try:
    print(f"Checking for existing collection '{COLLECTION_NAME}'...")

    # 1. Check if the collection exists. If it does, delete it.
    if client.collection_exists(collection_name=COLLECTION_NAME):
        print("Collection found. Deleting old collection...")
        client.delete_collection(collection_name=COLLECTION_NAME)

    # 2. Create the new collection with the required parameters
    print("Creating new collection...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    )
    print("Collection created successfully.")
except Exception as e:
    # Catch any general errors during this process
    print(f"Collection setup error: {e}")
# ----------------------------------------------------

db = DelegatorClient(client, COLLECTION_NAME, openai_embedding)

print(f"\nIndexing prompts into collection...")
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