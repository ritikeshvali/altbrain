from pka_core import MultiCollectionPKA
import os
from dotenv import load_dotenv
import json

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
pka = MultiCollectionPKA(api_key)

print("=" * 80)
print("EMBEDDINGS DEBUG VIEWER")
print("=" * 80)

collections = pka.list_collections()
for coll_info in collections:
    coll_name = coll_info['name']
    print(f"\n📦 Collection: {coll_name}")
    print("-" * 80)
    
    try:
        coll = pka.chroma_client.get_collection(coll_name, embedding_function=pka.embedding_function)
        data = coll.get(include=["embeddings", "metadatas", "documents"])
        
        print(f"Total embeddings: {len(data['embeddings'])}")
        print(f"Embedding dimension: {len(data['embeddings'][0]) if data['embeddings'] else 'N/A'}")
        
        for i, emb in enumerate(data['embeddings']):
            source = data['metadatas'][i].get('source', 'unknown')
            doc_snippet = data['documents'][i][:80].replace('\n', ' ')
            
            print(f"\n  [{i+1}] Source: {source}")
            print(f"      ID: {data['ids'][i]}")
            print(f"      Snippet: {doc_snippet}...")
            print(f"      Embedding dims: {len(emb)}")
            print(f"      First 10 values: {[round(x, 4) for x in emb[:10]]}")
            
    except Exception as e:
        print(f"Error reading collection: {e}")

print("\n" + "=" * 80)