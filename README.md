# AltBrain - Your Second Mind

a RAG app that lets you ask questions about your documents using AI

## Quick Start

### Setup

**Backend:**
```bash
cd backend
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
python main.py
```

Backend runs on http://localhost:8000

**Frontend:**
```bash
cd frontend
npm install
npm start
```

Frontend runs on http://localhost:3000

## Features
- Upload PDFs, TXT, MD files to collections
- Semantic search with multiple collections
- Add Google Docs for temporary context
- GPT-4o powered answers with source citations
- Rate limiting and security added
