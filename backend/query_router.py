from openai import OpenAI
from typing import List, Dict, Optional, AsyncGenerator


class QueryRouter:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    async def route_query(self, query: str, available_collections: List[str]) -> List[str]:
        system_prompt = """You are a query router. Given a query and available collections, determine which to search.

Collections:
- professional: SWE experience, work, projects
- academic: Education, research, coursework
- complete: Full profile
- mba: MBA apps, leadership, essays

Respond with ONLY comma-separated collection names."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=100,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\nAvailable: {', '.join(available_collections)}\nWhich collections?"}
            ]
        )
        
        selected = [c.strip() for c in response.choices[0].message.content.split(',')]
        valid = [c for c in selected if c in available_collections]
        return valid if valid else ["complete"]
    
    async def synthesize_answer(
        self,
        query: str,
        persistent_context: List[Dict],
        temp_context: Optional[Dict] = None
    ) -> AsyncGenerator[str, None]:
        context_parts = []
        
        for i, result in enumerate(persistent_context, 1):
            context_parts.append(f"[Source {i}: {result['source']} - {result['collection']}]\n{result['content']}")
        
        if temp_context:
            context_parts.append(f"[Temporary: Google Doc]\n{temp_context['content']}")
        
        context = "\n\n".join(context_parts)
        
        system_prompt = """Answer questions using provided context only. Cite sources with [Source N]. Be concise but comprehensive."""

        stream = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2000,
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content