# NutriManas-AI-Food-Quality-Nutrition-Analyzer
Upload any food photo → Get instant nutrition facts, quality analysis, and personalised health recommendations — powered by a 5-agent AI pipeline built specifically for Indian cuisine
NutriManas is a full-stack AI application that combines Computer Vision, Deep Learning, Retrieval-Augmented Generation (RAG), and Large Language Models into a single seamless pipeline. Point your camera at any food — from a plate of biryani to a raw guava — and get a complete nutritional breakdown with health guidance tailored to your personal profile in under 60 seconds.

---

## 🎯 Why I Built This

Most nutrition apps rely on a generic food database and manual logging. They fail with Indian cuisine because:
- Indian dishes have huge variation in ingredients and preparation styles
- Standard food databases (like USDA) have poor coverage for regional Indian food
- Calorie estimates for hand-portions and thalis are wildly inaccurate

NutriManas solves this with a purpose-built AI stack that **understands Indian food** — from identifying egg masala vs butter chicken to pulling nutrition from a curated Indian food knowledge base with 1,500+ entries.

---

## 🏗️ Architecture Overview

```
User uploads photo
        │
        ▼
┌─────────────────────────────────────────┐
│           Agent 1: Food Classifier       │
│  YOLOv8 (detect) → MobileNetV3 (classify│
│  80 Indian classes) → Gemini Vision      │
│  fallback (anything else)                │
└─────────────────────┬───────────────────┘
                      │ food_name + ingredients
                      ▼
┌─────────────────────────────────────────┐
│        Agent 2: Nutrition Calculator     │
│  pgvector RAG on Indian Food KB (1,500+ │
│  dishes) → USDA API fallback             │
└─────────────────────┬───────────────────┘
                      │ macros + micros
                      ▼
┌─────────────────────────────────────────┐
│         Agent 3: Quality Analyzer        │
│  Isolation Forest anomaly detection →   │
│  freshness score + contamination flag    │
└─────────────────────┬───────────────────┘
                      │ quality score
                      ▼
┌─────────────────────────────────────────┐
│        Agent 4: Health Recommender       │
│  Random Forest (user profile + nutrition │
│  → health risk) + Ollama llama3.1:8b    │
│  (personalised advice)                   │
└─────────────────────┬───────────────────┘
                      │ recommendations
                      ▼
┌─────────────────────────────────────────┐
│          Agent 5: LLM Explainer          │
│  Ollama llama3.1:8b synthesises all     │
│  outputs into friendly natural language  │
└─────────────────────┬───────────────────┘
                      │
                      ▼
              Full Analysis Result
        (nutrition + quality + health tips)
```

All agents are orchestrated by **LangGraph** — a stateful graph framework for multi-agent pipelines.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Next.js 14 (App Router) | React SSR framework |
| **UI** | TailwindCSS + shadcn/ui + Lucide Icons | Styling & components |
| **Backend** | FastAPI (Python) | REST API server |
| **Agent Framework** | LangGraph | Multi-agent orchestration |
| **Object Detection** | YOLOv8 (Ultralytics) | Detect food regions in image |
| **Food Classification** | MobileNetV3 (PyTorch) fine-tuned on 80 Indian food classes | Fast Indian dish identification |
| **Vision AI Fallback** | Google Gemini Flash Vision | Identify any food outside the 80 trained classes |
| **Nutrition RAG** | Supabase pgvector + nomic-embed-text | Vector similarity search on Indian Food KB |
| **Nutrition Fallback** | USDA FoodData Central API | Non-Indian foods + rare dishes |
| **Anomaly Detection** | scikit-learn Isolation Forest | Food freshness & contamination scoring |
| **Health ML** | scikit-learn Random Forest | Map user profile + nutrition → health risk |
| **LLM (local)** | Ollama llama3.1:8b | Health tips + natural language explanation |
| **Auth + Database** | Supabase | User auth, scan history, health profiles |

---

## 🤖 5 AI Agents — Deep Dive

### Agent 1 — Food Classifier
The most complex agent in the pipeline. Uses a **3-tier identification strategy**:

1. **YOLOv8** scans the image for food objects and returns bounding boxes with class labels
2. **MobileNetV3** (fine-tuned on 80 Indian food classes using transfer learning) classifies the detected region with a confidence score
3. If Gemini's answer is available → it overrides as the final identification. **Google Gemini Flash Vision** is called for every scan to ensure maximum accuracy — it can identify any food globally, not just the 80 trained classes

This hybrid approach means the app handles everything from *dal makhani* to *guava* to *marinated chicken* reliably.

### Agent 2 — Nutrition Calculator (RAG)
Uses **Retrieval-Augmented Generation** to fetch nutrition data:

1. Converts the food name into a vector embedding using `nomic-embed-text`
2. Runs a **pgvector similarity search** on the Indian Food Knowledge Base (1,500+ curated entries stored in Supabase)
3. For complex dishes (e.g. "chicken biryani"), decomposes into components and aggregates nutrition
4. Falls back to the **USDA FoodData Central API** for non-Indian foods or low-confidence KB matches
5. Returns: calories, protein, carbs, fats, fibre, vitamins, minerals per 100g and estimated portion

### Agent 3 — Quality Analyzer
Analyses food safety and freshness:

1. Extracts colour, texture, and statistical features from the image using OpenCV
2. Runs **Isolation Forest** (unsupervised anomaly detection) trained on normal food feature distributions
3. Outputs a **freshness score (0–100)** and flags potential contamination, overcooking, or spoilage
4. Uses `llama3.1:8b` to generate a 2-sentence human-readable quality assessment

### Agent 4 — Health Recommender
Personalised health advice based on who you are and what you ate:

1. Combines user health profile (age, weight, BMI, activity level, health conditions) with the nutrition output
2. **Random Forest classifier** predicts health risk category (low / moderate / high) for this meal
3. `llama3.1:8b` generates specific personalised recommendations — e.g. "Given your diabetes risk, the 62g of carbs in this portion exceeds your single-meal target"

### Agent 5 — LLM Explainer
The final summarisation agent:

- `llama3.1:8b` receives the structured outputs from all 4 preceding agents
- Writes a clear, friendly 3–5 sentence summary a non-technical user can immediately understand
- Highlights the most important finding (e.g. high sodium, excellent protein, freshness concern)

---

## ✨ Features

- **📸 Photo Upload** — drag-and-drop or click to upload any food photo (JPG, PNG, WEBP)
- **🍛 Indian Food Specialised** — MobileNetV3 trained on 80 Indian dish classes including biryani, dal makhani, palak paneer, chole bhature, vada pav and more
- **🌍 Universal Food Recognition** — Gemini Vision handles anything outside the 80 classes (fruits, international dishes, snacks)
- **📊 Full Nutrition Breakdown** — calories, protein, carbs, fat, fibre, vitamins, minerals with estimated portion weight
- **🔬 Quality & Freshness Score** — computer vision anomaly detection for food safety
- **❤️ Personalised Health Tips** — recommendations adapt to your age, BMI, health conditions, and goals
- **🗂️ Scan History** — all past analyses saved to your account with Supabase
- **👤 Health Profile** — set your dietary goals and health conditions once, get tailored advice every scan
- **🔐 Auth** — email/password signup via Supabase Auth

---

## 💡 Key Design Decisions

**Why MobileNetV3 + Gemini instead of just Gemini?**
MobileNetV3 runs locally in milliseconds at zero cost — it handles the 80 most common Indian dishes instantly. Gemini Vision handles everything else with high accuracy. This hybrid approach is faster and cheaper than calling Gemini for every single request.

**Why RAG for nutrition instead of just USDA?**
The USDA database has almost no Indian food. A curated Indian Food Knowledge Base with 1,500+ dishes was ingested into Supabase with pgvector embeddings. Vector similarity search retrieves the closest nutritional match by dish semantics, not exact name matching — so "shahi paneer" still retrieves correct data even if the KB entry is "paneer in cream gravy".

**Why LangGraph for orchestration?**
LangGraph allows the pipeline to be expressed as a directed graph with shared state. Each agent reads from and writes to a central `FoodAnalysisState` object. This makes it easy to add, remove, or swap agents without breaking the pipeline.

**Why Ollama for LLMs?**
Ollama runs `llama3.1:8b` fully locally — no API costs, no data leaving the machine, no latency from network calls for the LLM steps.

---

## 📁 Project Structure

```
NutriManas/
├── frontend/               # Next.js 14 app
│   └── src/
│       ├── app/            # App Router pages (home, scan, profile, history)
│       ├── components/     # UI components (Navbar, ResultsDisplay, etc.)
│       └── lib/            # Supabase client, API helpers
├── backend/                # FastAPI + LangGraph
│   ├── agents/
│   │   ├── food_classifier.py      # MobileNetV3 + Gemini Vision
│   │   ├── rag_nutrition_agent.py  # Indian KB RAG + USDA fallback
│   │   ├── quality_analyzer.py     # Isolation Forest
│   │   ├── health_recommender.py   # Random Forest + Ollama
│   │   └── llm_explainer.py        # Ollama summary
│   ├── models/             # Trained model weights (.pth, .joblib) — not in git
│   ├── pipeline.py         # LangGraph orchestration
│   ├── main.py             # FastAPI app
│   ├── config.py
│   └── requirements.txt
## Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.ai) installed locally

### 1. Pull Ollama Models
```bash
ollama pull llama3.1:8b
```

### 2. Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
cp .env.example .env        # Fill in keys (see below)
uvicorn main:app --port 8002
```

### 3. Frontend
```bash
cd frontend
npm install
npm run dev
```

### 4. Open
- Frontend: http://localhost:3000
- Backend API docs: http://localhost:8002/docs

---

## Environment Variables

Create `backend/.env` from `backend/.env.example`:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key
USDA_API_KEY=your_usda_api_key          # free at api.nal.usda.gov
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TEXT_MODEL=llama3.1:8b
GEMINI_API_KEY=your_gemini_api_key      # free at aistudio.google.com
```

---

## Supabase Tables

```sql
-- Scan history
create table scans (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id),
  food_name text,
  result jsonb,
  image_url text,
  created_at timestamptz default now()
);

-- User health profile
create table profiles (
  id uuid primary key references auth.users(id),
  age int,
  gender text,
  weight_kg float,
  height_cm float,
  bmi float,
  activity_level text,
  health_goal text,
  health_conditions text[],
  updated_at timestamptz default now()
);

---

## 🗺️ Roadmap

- [ ] Mobile app (React Native)
- [ ] Barcode scanner for packaged foods
- [ ] Meal planning and weekly nutrition summary
- [ ] Multi-language support (Hindi, Tamil, Telugu)
- [ ] Fine-tune MobileNetV3 on more regional Indian dishes (South Indian, Bengali, Gujarati)
- [ ] Share scan results as image cards

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with ❤️ for Indian food and AI.*
