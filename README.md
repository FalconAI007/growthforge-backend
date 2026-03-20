# GrowthForge Media — Backend API (Chanakya)

**Live API:** https://growthforge-api.onrender.com
**Health Check:** https://growthforge-api.onrender.com/health
**Website Repo:** https://github.com/FalconAI007/growthforge-website
**Dashboard Repo:** https://github.com/FalconAI007/growthforge-dashboard

---

## Project Overview

This is the Flask backend powering Chanakya — the AI sales agent for GrowthForge Media. It handles all AI processing, conversation management, lead storage, and Calendly booking detection across a 6-layer intelligent pipeline.

Every layer — intent detection, response generation, lead scoring, booking detection — uses GPT-4o-mini with no hardcoded rules, no decision trees, and no fixed keyword matching.

---

## Screenshots

- `screenshot-render-live.png` — Render dashboard showing service live with green deploy status
- `screenshot-pinecone-index.png` — Pinecone dashboard showing `growthforge` index with 88 vectors
- `screenshot-supabase-tables.png` — Supabase table editor showing leads, conversations, sessions tables

---

## Repository Structure

```
growthforge-backend/
├── api.py                              # Flask app — all routes and Calendly logic
├── core/
│   ├── __init__.py
│   ├── intent_layer.py                 # GPT intent detection (4-dimension reasoning)
│   ├── router.py                       # Conversation orchestration and stage engine
│   ├── lead_capture.py                 # Lead management wrapper
│   └── database.py                     # All Supabase operations
├── scripts/
│   ├── __init__.py
│   ├── ingest.py                       # Pinecone ingestion — run once to populate
│   └── query.py                        # RAG retrieval + response generation
├── data/
│   ├── services.txt                    # Services knowledge
│   ├── pricing.txt                     # Pricing packages
│   ├── faq.txt                         # FAQs
│   ├── case_studies.txt                # Client results
│   ├── agency.txt                      # Agency overview
│   ├── facebook_ads_service.txt        # Facebook Ads deep-dive
│   └── client_onboarding_process.txt   # Onboarding steps
├── render.yaml                         # Render deployment config
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Features

- **GPT Intent Detection** — Classifies every message into 9 intents: PRICING, SERVICE, CASE, ROI, ONBOARDING, FAQ, OBJECTION, GENERAL, OUT_OF_SCOPE. Uses a 4-dimension reasoning framework — relationship stage, emotional state, business relevance, user need.
- **RAG with Pinecone** — 88 vectors stored using OpenAI embeddings (1536 dimensions, cosine similarity). Retrieves the most semantically relevant knowledge before generating every response.
- **Dual RAG Usage** — The same retrieval pipeline feeds both the intent detection layer and the response generation layer — both are grounded in real knowledge.
- **Conversation Stage Engine** — Tracks visitor journey across AWARENESS → CONSIDERATION → DECISION → OBJECTION_HANDLING with intelligent transitions based on message count and intent patterns.
- **GPT Lead Scoring** — Every conversation scored 1-10 by GPT reading the full context. No hardcoded thresholds — pure semantic analysis of buying signals and engagement quality.
- **Intelligent Calendly Trigger** — GPT booking intent classifier distinguishes genuine confirmations from general agreement before showing the booking button.
- **Supabase Persistent Storage** — All sessions, conversations, and leads stored in PostgreSQL — surviving server restarts and powering the dashboard.
- **AI Conversation Summary** — GPT generates a one-line summary after each exchange: *"Juice shop owner asking about Growth Package, ready to book."*

---

## Mentor Suggestions Implemented

**Cosine Similarity for RAG**
Mentor recommended cosine similarity for vector retrieval. Implemented via Pinecone with `metric: cosine` on a 1536-dimension OpenAI embedding index. This ensures accurate retrieval even when the visitor's exact words differ from the knowledge base.

**RAG for Both Intent and Response**
Mentor suggested the knowledge base should feed both layers — not just the response. We pass retrieved context to the intent classifier as background knowledge and separately to the response generator. Both layers are grounded in accurate data.

---

## Why These Technologies

**Pinecone over ChromaDB**
We built the prototype with ChromaDB. The problem: ChromaDB stores data on disk inside the Render container which gets wiped on every redeploy — all vectors were lost on each deployment. Pinecone is a managed cloud vector database that persists independently. It also eliminates infrastructure management entirely.

**Supabase over JSON file storage**
The prototype used a `leads.json` file on the server. Render's filesystem is ephemeral — every restart wiped all leads. Supabase gives us managed PostgreSQL with zero configuration and persistent storage that survives every server event.

**OpenAI Embeddings over sentence-transformers**
We initially used `all-MiniLM-L6-v2` locally. On Render's free tier (512MB RAM) this caused out-of-memory crashes — the model alone consumes ~400MB on startup. Switching to OpenAI's `text-embedding-3-small` API eliminated the RAM issue completely while improving retrieval quality.

**n8n not included in MVP**
n8n is excellent for post-lead automation — email sequences, CRM pushes, Slack notifications. We kept it out of the MVP deliberately: adding n8n would introduce another hosted service requiring configuration and maintenance. The priority was making the core AI pipeline robust and production-ready first. n8n integration is the first item on the post-hackathon roadmap.

---

## Architecture

```
Website (Vercel)
    │
    ▼
Flask API (Render)
    ├── Intent Layer        GPT-4o-mini — 4-dimension reasoning
    ├── RAG Pipeline        OpenAI Embeddings + Pinecone cosine search
    ├── Response Gen        GPT-4o-mini + Chanakya system prompt
    ├── Stage Engine        AWARENESS → CONSIDERATION → DECISION
    ├── Lead Scoring        GPT-4o-mini — 1-10 full conversation analysis
    ├── Calendly Check      GPT-4o-mini — booking intent classifier
    └── Supabase            leads / conversations / sessions tables
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/chat` | POST | Main chat — processes message, returns response |
| `/capture-lead` | POST | Save or update lead contact details |
| `/leads` | GET | All leads for dashboard |
| `/conversations/<session_id>` | GET | Full conversation transcript |
| `/health` | GET | Health check |

---

## Setup and Installation

**Prerequisites:** Python 3.10+, OpenAI API key, Pinecone account, Supabase project

```bash
# Clone
git clone https://github.com/FalconAI007/growthforge-backend.git
cd growthforge-backend

# Virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Fill in your API keys

# Populate Pinecone (run once)
python scripts/ingest.py

# Run locally
python api.py
```

**Environment Variables:**

```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX=growthforge
DATABASE_URL=your_supabase_session_pooler_url
CALENDLY_LINK=https://calendly.com/your-link
FLASK_ENV=production
```

**Supabase Tables — run in SQL Editor:**

```sql
CREATE TABLE leads (
    id SERIAL PRIMARY KEY,
    session_id TEXT UNIQUE NOT NULL,
    name TEXT, email TEXT, phone TEXT, business TEXT,
    intent TEXT, stage TEXT, conversation_summary TEXT,
    lead_score INTEGER DEFAULT 1,
    conversation_history JSONB DEFAULT '[]',
    status TEXT DEFAULT 'new',
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    last_updated TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    stage TEXT DEFAULT 'AWARENESS',
    last_intent TEXT,
    message_count INTEGER DEFAULT 0,
    meaningful_message_count INTEGER DEFAULT 0,
    cta_shown BOOLEAN DEFAULT FALSE,
    cta_shown_count INTEGER DEFAULT 0,
    tone TEXT DEFAULT 'neutral',
    email_provided BOOLEAN DEFAULT FALSE,
    phone_provided BOOLEAN DEFAULT FALSE,
    objection_count INTEGER DEFAULT 0,
    intents_seen TEXT DEFAULT '',
    lead_score INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_updated TIMESTAMPTZ DEFAULT NOW()
);
```

**Render Deployment:**
1. Push repo to GitHub
2. New Web Service on Render → connect repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn api:app --bind 0.0.0.0:$PORT`
5. Add environment variables
6. Deploy

---

## Future Ideas

- **n8n Workflow Automation** — Automate follow-up emails, CRM pushes, and Slack alerts triggered by lead score thresholds
- **CRM Integration** — Push hot leads directly to Airtable or HubSpot with full transcript attached
- **Learning Loop** — Track which conversations converted to bookings and improve scoring accuracy over time
- **Multi-tenant Architecture** — White-label for marketing agencies to deploy for their own clients
- **WhatsApp and Instagram DM** — Extend Chanakya beyond the website to meet leads on their preferred platforms

---

## Novelty

Existing chatbot platforms use scripted flows or basic NLP. Chanakya uses GPT reasoning at every layer with no fixed paths. Every decision is made by reading the full conversation context — intent, score, booking — exactly as a human sales rep would think through it.

---

## Tech Stack

`Python 3.14` `Flask` `Gunicorn` `OpenAI GPT-4o-mini` `OpenAI text-embedding-3-small` `Pinecone` `Supabase PostgreSQL` `LangChain` `Render`
