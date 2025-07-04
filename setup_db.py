import os
import chromadb
from chromadb.utils import embedding_functions

from dotenv import load_dotenv
load_dotenv()

EVENT_DIR = "data/events"
CHROMA_DB_DIR = ".chroma_db"

db_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

openai_embedding = embedding_functions.OpenAIEmbeddingFunction(
    api_key =os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small",
)

def init_collection():
    """Initialize the ChromaDB collection for events."""
    
    try:
        collection = db_client.get_collection("event_titles", embedding_function=openai_embedding)
        return collection
    
    except:
        collection = db_client.create_collection("event_titles", embedding_function=openai_embedding,metadata={"hnsw:space": "cosine"})

        event_files = [f for f in os.listdir(EVENT_DIR) if f.endswith('.md')]

        docs = []
        ids = []
        metadatas = []
        for file in event_files:
            title = ' '.join(file.split('.')[0].split('_')[:-1])
            page_id = file.split('.')[0]

            docs.append(title)
            ids.append(page_id)
            metadatas.append({"page_id": page_id, "title": title})

        batch_size = 50
        for i in range(0, len(docs), batch_size):
            collection.add(
                documents=docs[i:i + batch_size],
                ids=ids[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size]
            )

        return collection
    
if __name__ == "__main__":
    collection = init_collection()
    print(f"Initialized collection with {collection.count()} documents.")
    print("Collection metadata:", collection.get_metadata())
    print("Collection info:", collection.get_info())
    print("List of collections:", db_client.list_collections())