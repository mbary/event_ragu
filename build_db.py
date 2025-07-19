import os
from dotenv import load_dotenv
load_dotenv()

import asyncio
from pydantic import BaseModel, Field

import instructor
from openai import AsyncOpenAI
import chromadb
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from tqdm.asyncio import tqdm

EVENT_DIR = "data/events"
CHROMA_DB_DIR = ".chroma_db"
COLLECTION_NAME = "event_semantic_summaries"
MODEL_NAME = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

class SemanticExtraction(BaseModel):
    """Distilled, semantically rich summary of an event for embedding."""
    title: str = Field(description="The official, clean title of the event.")
    summary: str = Field(description="A concise, 1-3 sentence summary of what the event is about, its theme and purpose. This Should NOT include logistics like dates, times or prices.")


client = instructor.from_openai(AsyncOpenAI())
semaphore = asyncio.Semaphore(5)

async def distill_event_content(filepath: str) -> SemanticExtraction:
    page_id = os.path.basename(filepath).replace('.md', '')
    async with semaphore:
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()

        system_prompt = """You are a highly efficient data distiller. Your sole purpose is to read raw text and extract its core semantic essence into a structured format. 

        **CRITICAL INSTRUCTIONS:**
        1.  Extract the official, clean **title** of the event.
        2.  Create a concise, 1-3 sentence **summary** that describes WHAT the event is about, its theme, and its purpose.
        3.  **YOU MUST IGNORE ALL LOGISTICAL INFORMATION.** Do not include dates, times, URLs, contact info, registration details, prices, or boilerplate text in the summary. Focus ONLY on the thematic description.
        4. The summary **MUST** be in Polish language.
        """
        
        user_prompt = f"""
        Please distill the following event content into a structured format with a clean title and a concise summary.
        ```
        {content}
        ```
        """
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_model=SemanticExtraction,
                temperature=0.2)
            
            return page_id, response
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return page_id, None
        
async def generate_event_summaries(event_dir: str):
    """Generate semantic summaries for all event files in the directory."""
    tasks = []
    for filename in os.listdir(event_dir):
        if filename.endswith('.md'):
            filepath = os.path.join(event_dir, filename)
            tasks.append(distill_event_content(filepath))
    results = await tqdm.gather(*tasks, desc="Distilling event files")
    return {page_id: summary for page_id, summary in results if summary}

async def main():
    event_dir = "data/events"
    summaries = await generate_event_summaries(event_dir)
    return summaries
    

def create_embedding_collection(summaries_dict: dict[str, SemanticExtraction]):
    print("\nInitializing ChromaDB client...")
    db_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    openai_embedding_func = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=EMBEDDING_MODEL,
    )

    if COLLECTION_NAME in [c.name for c in db_client.list_collections()]:
        print(f"Deleting existing collection: '{COLLECTION_NAME}'")
        db_client.delete_collection(name=COLLECTION_NAME)

    print(f"Creating new collection: '{COLLECTION_NAME}'")
    collection = db_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_embedding_func,
        metadata={"hnsw:space": "cosine"}
    )

    docs_to_embed = []
    ids = []
    metadatas = []
    for page_id, summary in summaries_dict.items():
        document_for_embedding = f"Tytuł: {summary.title}\n\nOpis: {summary.summary}"
        docs_to_embed.append(document_for_embedding)
        ids.append(page_id)
        metadatas.append({"page_id": page_id, "title": summary.title})

    print(f"Adding {len(docs_to_embed)} documents to the collection...")
    batch_size = 100
    for i in range(0, len(docs_to_embed), batch_size):
        collection.add(
            documents=docs_to_embed[i:i + batch_size],
            ids=ids[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )
    
    print("\n--- Database Creation Complete ---")
    print(f"Collection '{collection.name}' initialized with {collection.count()} documents.")

async def main_build():
    summaries = await generate_event_summaries(EVENT_DIR)
    if not summaries:
        print("No summaries were generated. Exiting.")
        return
    create_embedding_collection(summaries)

if __name__ == "__main__":
    asyncio.run(main_build())