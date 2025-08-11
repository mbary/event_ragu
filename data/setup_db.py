import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_DIR = ".chroma_db"
COLLECTION_NAME = "event_semantic_summaries"
EMBEDDING_MODEL = "text-embedding-3-small"

def init_collection() -> chromadb.Collection:
    """
    Connects to the persistent ChromaDB client and returns the existing
    collection of event summaries.
    
    This function does NOT create or modify the collection. It assumes
    the database has already been built by `build_database.py`.
    """
    db_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    openai_embedding_func = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=EMBEDDING_MODEL,
    )
    
    try:
        collection = db_client.get_collection(
            name=COLLECTION_NAME, 
            embedding_function=openai_embedding_func
        )
        print(f"Successfully connected to existing collection '{COLLECTION_NAME}' with {collection.count()} documents.")
        return collection
    except Exception as e:
        print(f"FATAL ERROR: Could not connect to ChromaDB collection '{COLLECTION_NAME}'.")
        print("Please ensure you have run the `build_database.py` script first.")
        print(f"Original error: {e}")
        raise

if __name__ == "__main__":
    print("Attempting to initialize collection...")
    try:
        collection = init_collection()
    except Exception as e:
        print("\nVerification failed.")