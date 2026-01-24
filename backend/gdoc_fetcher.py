import re
import httpx
from typing import Optional, Dict


class GoogleDocFetcher:
    def __init__(self):
        self.client = httpx.AsyncClient()
    
    def extract_doc_id(self, url: str) -> Optional[str]:
        patterns = [r'/document/d/([a-zA-Z0-9-_]+)', r'id=([a-zA-Z0-9-_]+)']
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    async def fetch_content(self, url: str) -> str:
        doc_id = self.extract_doc_id(url)
        if not doc_id:
            raise ValueError("Invalid Google Docs URL")
        
        export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
        response = await self.client.get(export_url, follow_redirects=True)
        
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Failed to fetch document: {response.status_code}")
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
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
    
    async def fetch_and_embed(self, url: str) -> Dict:
        text = await self.fetch_content(url)
        chunks = self.chunk_text(text)
        combined = "\n\n".join(chunks)
        
        return {
            "content": combined,
            "source": "Google Doc",
            "url": url,
            "chunk_count": len(chunks)
        }
    
    async def close(self):
        await self.client.aclose()