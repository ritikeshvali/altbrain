import sqlite3
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import PyPDF2
from openai import OpenAI
from chromadb import Client
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions

RELEVANCE_THRESHOLD = 0.75

class MultiCollectionPKA:
    def __init__(self, api_key: str, db_path: str = "pka.db"):
        self.api_key = api_key
        self.db_path = db_path
        self.openai_client = OpenAI(api_key=api_key)
        
        self.chroma_client = Client(Settings(
            persist_directory="./chroma_db",
            anonymized_telemetry=False
        ))
        
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )
        
        self._init_db()
        self._ensure_default_collections()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                uploaded_at TEXT NOT NULL,
                FOREIGN KEY (collection_id) REFERENCES collections (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _ensure_default_collections(self):
        defaults = [
            ("professional", "Software engineering, projects, work experience"),
            ("academic", "Education, research, coursework"),
            ("complete", "Complete profile - all information"),
            ("mba", "MBA applications, essays, leadership")
        ]
        
        for name, desc in defaults:
            try:
                self.create_collection(name, desc)
            except:
                pass
    
    def create_collection(self, name: str, description: str):
        self.chroma_client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO collections (name, description, created_at) VALUES (?, ?, ?)",
                (name, description, datetime.now().isoformat())
            )
            conn.commit()
        except sqlite3.IntegrityError:
            pass
        finally:
            conn.close()
    
    def list_collections(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name, description, created_at FROM collections")
        rows = cursor.fetchall()
        conn.close()
        
        collections = []
        for name, desc, created in rows:
            try:
                coll = self.chroma_client.get_collection(name, embedding_function=self.embedding_function)
                doc_count = coll.count()
            except:
                doc_count = 0
            
            collections.append({
                "name": name,
                "description": desc,
                "created_at": created,
                "document_count": doc_count
            })
        
        return collections
    
    def delete_collection(self, name: str):
        try:
            self.chroma_client.delete_collection(name)
        except:
            pass
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM collections WHERE name = ?", (name,))
        conn.commit()
        conn.close()
    
    def chunk_text(self, text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    def extract_text(self, file_path: str) -> str:
        path = Path(file_path)
        
        if path.suffix.lower() == '.pdf':
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        elif path.suffix.lower() in ['.txt', '.md']:
            return path.read_text(encoding='utf-8')
        elif path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Images: store filename as text (OCR would require additional library)
            return f"Image file: {path.name}"
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    
    def ingest_document(self, file_path: str, collection_name: str) -> Dict:
        path = Path(file_path)
        text = self.extract_text(file_path)
        chunks = self.chunk_text(text)
        
        collection = self.chroma_client.get_collection(
            collection_name,
            embedding_function=self.embedding_function
        )
        
        doc_id = f"{collection_name}_{path.stem}_{datetime.now().timestamp()}"
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        
        metadatas = [{
            "source": path.name,
            "collection": collection_name,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "upload_date": datetime.now().isoformat()
        } for i in range(len(chunks))]
        
        collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM collections WHERE name = ?", (collection_name,))
        coll_id = cursor.fetchone()[0]
        cursor.execute(
            "INSERT INTO documents (collection_id, filename, filepath, chunk_count, uploaded_at) VALUES (?, ?, ?, ?, ?)",
            (coll_id, path.name, str(path.absolute()), len(chunks), datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        
        return {"chunks": len(chunks), "collection": collection_name}
    
    def search_multi_collection(self, query: str, collections: List[str], n_results: int = 10) -> List[Dict]:
        all_results = []
        
        for coll_name in collections:
            try:
                collection = self.chroma_client.get_collection(coll_name, embedding_function=self.embedding_function)
                results = collection.query(query_texts=[query], n_results=n_results)
                
                if results['documents'] and results['documents'][0]:
                    for i, doc in enumerate(results['documents'][0]):
                        all_results.append({
                            "content": doc,
                            "source": results['metadatas'][0][i].get('source', 'Unknown'),
                            "collection": coll_name,
                            "distance": results['distances'][0][i],
                            "metadata": results['metadatas'][0][i]
                        })
            except:
                continue
        
        # Sort by distance (lower is better) and content length (longer is better)
        all_results.sort(key=lambda x: (x['distance'], -len(x['content'])))
        
        relevant_results = [r for r in all_results if r['distance'] < RELEVANCE_THRESHOLD]
        
        return relevant_results[:n_results]
    
    def list_documents(self, collection_name: str) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT d.filename, d.chunk_count, d.uploaded_at
            FROM documents d
            JOIN collections c ON d.collection_id = c.id
            WHERE c.name = ?
        """, (collection_name,))
        rows = cursor.fetchall()
        conn.close()
        
        return [{"filename": r[0], "chunks": r[1], "uploaded_at": r[2]} for r in rows]