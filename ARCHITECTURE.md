# Architecture (Python FastAPI + Transformers)

This document describes the Python-based, OpenAI-compatible inference server for Qwen3-4B-Instruct, designed to power an **AI-powered marketplace intelligence system**.

## System Purpose

This inference server serves as the AI backend for a smart marketplace platform where:
- Suppliers can register and list products
- Users query product availability and get AI recommendations
- Products are matched based on user location to find nearest suppliers
- AI assistant helps with natural language product discovery

Key source files
- Server entry: [main.py](main.py)
- Inference engine: [Python.class Engine](main.py:464)
- Endpoints: Health [Python.app.get()](main.py:577), Chat Completions [Python.app.post()](main.py:591), Cancel [Python.app.post()](main.py:792)
- Streaming + resume: [Python.class _SSESession](main.py:435), [Python.class _SessionStore](main.py:449), [Python.class _SQLiteStore](main.py:482), [Python.function chat_completions](main.py:591)
- Local run (uvicorn): [Python.main()](main.py:807)
- Configuration template: [.env.example](.env.example)
- Dependencies: [requirements.txt](requirements.txt)

Model target (default)
- Hugging Face: unsloth/Qwen3-4B-Instruct-2507 (Transformers, text-only instruct model)
- Overridable via environment variable: MODEL_REPO_ID

**Deprecated features**: Multimodal parsing (images/videos) and KTP OCR endpoint are deprecated as of migration to text-only model. Code remains for reference but is non-functional.

## Overview

The server exposes an OpenAI-compatible endpoint for chat completions:
- **Text-only prompts** (primary use case for marketplace AI assistant)
- Non-streaming JSON responses
- Streaming via Server-Sent Events (SSE) with resumable delivery using Last-Event-ID
- Resumability is achieved with an in‑memory ring buffer and optional SQLite persistence

## Components

1) FastAPI application
- Instantiated in [Python.main module](main.py:541) and endpoints mounted at:
  - Health: [Python.app.get()](main.py:577)
  - Chat Completions (non-stream + SSE): [Python.app.post()](main.py:591) - **Primary endpoint for marketplace AI**
  - Manual cancel (custom): [Python.app.post()](main.py:792)
  - ~~KTP OCR: [Python.app.post()](main.py:1310)~~ - **DEPRECATED** (requires multimodal model)
- CORS is enabled for simplicity.

2) Inference Engine (Transformers)
- Class: [Python.class Engine](main.py:464)
- Loads:
  - Processor: AutoProcessor(trust_remote_code=True)
  - Model: AutoModelForCausalLM (device_map, dtype configurable via env)
- Core methods:
  - Text-only generate: [Python.function infer](main.py:326)
  - Streaming generate (iterator): [Python.function infer_stream](main.py:375)

3) ~~Multimodal preprocessing~~ - **DEPRECATED**
- ~~Images/Videos processing~~ - Code remains but is non-functional with text-only model
- For marketplace use case, all interactions are text-based:
  - Product queries: "laptop gaming Jakarta"
  - Recommendations: "laptop programming 10 juta"
  - Location queries: supplier location data passed as text in conversation context

4) SSE streaming with resume
- Session objects:
  - [Python.class _SSESession](main.py:435): ring buffer, condition variable, producer thread reference, cancellation event, listener count, and disconnect timer
  - [Python.class _SessionStore](main.py:449): in-memory map with TTL + GC
  - Optional persistence: [Python.class _SQLiteStore](main.py:482) for replaying chunks across restarts
- SSE id format: "session_id:index"
- Resume:
  - Client sends Last-Event-ID header (or query ?last_event_id=...) and the same session_id in the body
  - Server replays cached/persisted chunks after the provided index, then continues live streaming
- Producer:
  - Created on demand per session; runs generation in a daemon thread and pushes chunks into the ring buffer and SQLite (if enabled)
  - See producer closure inside [Python.function chat_completions](main.py:591)
- Auto-cancel on disconnect:
  - If all clients disconnect for CANCEL_AFTER_DISCONNECT_SECONDS (default 3600s), a timer signals cancellation via a stopping criteria in [Python.function infer_stream](main.py:375)

## Request flow

Non-streaming (POST /v1/chat/completions)
1. Validate input, load engine singleton via [Python.function get_engine](main.py:558)
2. Convert OpenAI-style messages to Qwen chat template via apply_chat_template
3. ~~Preprocess images/videos~~ - DEPRECATED (text-only model)
4. Generate with [Python.function infer](main.py:326)
5. Return OpenAI-compatible response (choices[0].message.content)

Streaming (POST /v1/chat/completions with "stream": true)
1. Determine session_id:
   - Use body.session_id if provided; otherwise generated server-side
2. Parse Last-Event-ID (or query ?last_event_id) to get last delivered index
3. Create/start or reuse producer thread for this session
4. StreamingResponse generator:
   - Replays persisted events (SQLite, if enabled) and in-memory buffer after last index
   - Waits on condition variable for new tokens
   - Emits "[DONE]" at the end or upon buffer completion
5. Clients can reconnect and resume by sending Last-Event-ID: "session_id:index"
6. If all clients disconnect, an auto-cancel timer can stop generation (configurable via env)

Manual cancel (POST /v1/cancel/{session_id})
- Custom operational shortcut to cancel an in-flight generation for a session id.
- This is not part of the legacy OpenAI Chat Completions spec (OpenAI’s newer Responses API defines cancel); it is provided for practical control.

KTP OCR (POST /ktp-ocr/)
- Specialized endpoint for Indonesian ID card (KTP) optical character recognition.
- Accepts multipart form-data with image file, extracts structured JSON data using multimodal inference.
- Returns standardized fields: nik, nama, tempat_lahir, tgl_lahir, jenis_kelamin, alamat (with nested fields), agama, status_perkawinan, pekerjaan, kewarganegaraan, berlaku_hingga.
- Uses custom prompt engineering for accurate structured extraction from Qwen3-VL model.
- Inspired by raflyryhnsyh/Gemini-OCR-KTP but adapted for local, self-hosted inference.

## Message and content mapping

Input format (OpenAI-like):
- "messages" list of role/content entries
- content can be:
  - string (text)
  - array of parts with "type":
    - "text": { text: "..."}
    - "image_url": { image_url: { url: "..." } } or { image_url: "..." }
    - "input_image": { b64_json: "..." } or { image: "..." }
    - "video_url": { video_url: { url: "..." } } or { video_url: "..." }
    - "input_video": { b64_json: "..." } or { video: "..." }

Conversion:
- [Python.function build_mm_messages](main.py:251) constructs a multimodal content list per message:
  - { type: "text", text: ... }
  - { type: "image", image: PIL.Image }
  - { type: "video", video: [PIL.Image frames] }

Template:
- Qwen apply_chat_template:
  - See usage in [Python.function infer](main.py:326) and [Python.function infer_stream](main.py:375)

## Configuration (.env)

See [.env.example](.env.example)
- PORT (default 3000)
- MODEL_REPO_ID (default "unsloth/Qwen3-4B-Instruct-2507")
- HF_TOKEN (optional)
- MAX_TOKENS (default 256)
- TEMPERATURE (default 0.7)
- MAX_VIDEO_FRAMES (default 16)
- DEVICE_MAP (default "auto")
- TORCH_DTYPE (default "auto")
- PERSIST_SESSIONS (default 0; set 1 to enable SQLite persistence)
- SESSIONS_DB_PATH (default sessions.db)
- SESSIONS_TTL_SECONDS (default 600)
- CANCEL_AFTER_DISCONNECT_SECONDS (default 3600; set 0 to disable)

## Error handling and readiness

- Health endpoint: [Python.app.get()](main.py:577)
  - Returns { ok, modelReady, modelId, error }
- Chat endpoint:
  - 400 for invalid messages or multimodal parsing errors
  - 503 when model failed to load
  - 500 for unexpected generation errors
- During first request, the model is lazily loaded; subsequent requests reuse the singleton

## Performance and scaling

- GPU recommended:
  - Set DEVICE_MAP=auto and TORCH_DTYPE=bfloat16/float16 if supported
- Reduce MAX_VIDEO_FRAMES to speed up video processing
- For concurrency:
  - FastAPI/Uvicorn workers and model sharing: typically 1 model per process
  - For high throughput, prefer multiple processes or a queueing layer

## Data and directories

- models/ contains downloaded model artifacts (implicitly created by Transformers cache); ignored by git
- tmp/ used transiently for video decoding (temporary files)

Ignored artifacts (see [.gitignore](.gitignore))
- Python: .venv/, __pycache__/, .cache/, etc.
- Large artifacts: models/, data/, uploads/, tmp/

## Streaming resume details

- Session store:
  - In-memory ring buffer for fast replay
  - Optional SQLite persistence for robust replay across process restarts
  - See GC in [Python.class _SessionStore](main.py:449) and [Python.method _SQLiteStore.gc](main.py:526)
- Limits:
  - Ring buffer stores ~2048 SSE events per session by default
  - If the buffer overflows before a client resumes and persistence is disabled, the earliest chunks may be unavailable
- End-of-stream:
  - Final chunk contains finish_reason: "stop"
  - "[DONE]" sentinel is emitted afterwards

## Marketplace Integration Plan

This inference server is designed to power an AI-powered marketplace platform. The following components need to be developed:

### 1. Database Schema (Planned)
- **Suppliers table**: id, name, business_name, location (lat/lng), address, contact, registration_date
- **Products table**: id, supplier_id, name, description, price, stock_quantity, category, tags
- **Users table**: id, name, email, location (lat/lng), ai_access_enabled, preferences
- **Conversations table**: id, user_id, session_id, created_at (for chat history)
- **Messages table**: id, conversation_id, role (user/assistant), content, timestamp

### 2. Marketplace API Endpoints (Planned)
- **Supplier Management**:
  - POST /api/suppliers/register - Register new supplier
  - POST /api/products - Add product listing
  - PUT /api/products/{id} - Update product (stock, price)
  - GET /api/suppliers/{id}/products - List supplier products

- **Product Search**:
  - GET /api/products/search?q={query}&location={lat,lng} - Traditional search
  - POST /api/products/ai-search - AI-powered search with natural language

- **AI Assistant Integration**:
  - POST /api/chat - Wrapper around /v1/chat/completions with context injection
  - Context injection: Pass product database results as system message
  - Location awareness: Calculate distance, sort by proximity
  - Example flow:
    1. User: "laptop gaming Jakarta budget 10 juta"
    2. Backend queries products table for: category="laptop", tags LIKE "%gaming%", location near Jakarta, price <= 10000000
    3. Inject results into system prompt: "Available products: [JSON array of matching products]"
    4. Send to /v1/chat/completions
    5. AI recommends from available inventory with reasons

### 3. Location-Aware Features (Planned)
- Geolocation distance calculation (Haversine formula)
- Sort products/suppliers by distance from user
- Multi-supplier comparison showing nearest options
- Delivery time estimates based on distance

### 4. Context Management Strategy
- Maintain conversation history per user session
- Inject product catalog context dynamically based on query
- Use session_id for resumable conversations
- Store conversation in database for analytics and personalization

### Current Status
- ✅ Inference server ready (/v1/chat/completions endpoint)
- ✅ Streaming and resume functionality
- ⏳ Database schema design
- ⏳ Marketplace API development
- ⏳ Frontend/UI development
- ⏳ Product catalog seeding and testing

## Future enhancements

- Redis persistence:
  - Add a Redis-backed store as a drop-in alongside SQLite
- Token accounting:
  - Populate usage prompt/completion/total tokens when model exposes tokenization costs
- Logging/observability:
  - Structured logs, request IDs, and metrics

## Migration notes

### From Node.js to Python (2025-10-23)
- All Node.js server files and scripts were removed (index.js, package*.json, scripts/)
- Migrated to Python FastAPI + Transformers stack
- The API remains OpenAI-compatible on /v1/chat/completions with resumable SSE and optional SQLite persistence

### From Multimodal to Text-Only (2025-11-13)
- Migrated from Qwen/Qwen3-VL-2B-Thinking (multimodal) to unsloth/Qwen3-4B-Instruct-2507 (text-only)
- **Deprecated features**: KTP OCR endpoint, image/video processing
- **New focus**: AI-powered marketplace intelligence system
- Multimodal code remains in codebase but is non-functional with current model
