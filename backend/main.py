from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import unquote
from pka_core import MultiCollectionPKA
from query_router import QueryRouter
from gdoc_fetcher import GoogleDocFetcher
import secrets
import re
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address

load_dotenv()

app = FastAPI(title="AltBrain - Your Second Mind")

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_credentials=False,
    allow_headers=["*"],
)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

pka = MultiCollectionPKA(api_key)
router = QueryRouter(api_key)
gdoc_fetcher = GoogleDocFetcher()

# File size limit (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

class AskRequest(BaseModel):
    query: str
    collections: List[str]
    gdoc_link: Optional[str] = None


class CreateCollectionRequest(BaseModel):
    name: str
    description: str


@app.get("/")
async def root():
    return {"status": "PKA API Running"}


@app.get("/collections")
async def list_collections():
    return {"collections": pka.list_collections()}


@app.post("/collections")
async def create_collection(req: CreateCollectionRequest):
    if not re.match(r'^[a-zA-Z0-9_-]{1,50}$', req.name):
        raise HTTPException(status_code=400, detail="Invalid collection name")
    pka.create_collection(req.name, req.description)
    return {"status": "created", "name": req.name}


@app.delete("/collections/{name}")
async def delete_collection(name: str):
    pka.delete_collection(name)
    return {"status": "deleted"}


@app.post("/ingest")
@limiter.limit("5/minute")
async def ingest_file(request: Request, file: UploadFile = File(...), collection: str = Form(...)):
    # File size check
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")
    
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    safe_filename = secrets.token_hex(8) + "_" + Path(file.filename).name
    file_path = upload_dir / safe_filename
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    try:
        result = pka.ingest_document(str(file_path), collection)
        return {
            "status": "success",
            "filename": file.filename,
            "collection": collection,
            "chunks": result["chunks"]
        }
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error: {e}", exc_info=True) 
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/ask")
@limiter.limit("10/minute")
async def ask_question(request: Request, req: AskRequest):
    try:
        persistent_results = pka.search_multi_collection(
            query=req.query,
            collections=req.collections,
            n_results=10
        )
        
        temp_context = None
        if req.gdoc_link:
            temp_context = await gdoc_fetcher.fetch_and_embed(req.gdoc_link)
        
        async def generate():
            async for chunk in router.synthesize_answer(
                query=req.query,
                persistent_context=persistent_results,
                temp_context=temp_context
            ):
                yield chunk
        
        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/documents/{collection}")
async def list_documents(collection: str):
    return {"collection": collection, "documents": pka.list_documents(collection)}


@app.delete("/documents/{collection}")
async def delete_document(collection: str, filename: str = Query(...)):
    # URL decode filename
    filename = unquote(filename)
    
    # Delete from ChromaDB
    try:
        coll = pka.chroma_client.get_collection(collection, embedding_function=pka.embedding_function)
        all_data = coll.get(include=["metadatas"])
        
        ids_to_delete = [
            all_data['ids'][i] 
            for i, meta in enumerate(all_data['metadatas'])
            if meta.get('source') == filename
        ]
        
        if ids_to_delete:
            coll.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} chunks")
    except Exception as e:
        print(f"ChromaDB delete error: {e}")
    
    # Delete from SQLite
    conn = sqlite3.connect(pka.db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE filename = ?", (filename,))
    deleted_rows = cursor.rowcount
    conn.commit()
    conn.close()
    
    print(f"Deleted from SQLite: {deleted_rows} rows")
    return {"status": "deleted", "filename": filename}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)