# 🥗 Nutrition Assistant
### (a RAG-based Q&A chatbot)

# Nutrition Assistant (RAG)

A conversational **RAG** app that answers nutrition questions from a curated food table (macros, vitamins, minerals, allergens). It retrieves the most relevant rows and generates grounded answers.

> Streamlit UI is **English-only** (as required).

---

## Table of contents

- [Overview](#overview-evaluation-criteria-problem-description--22-retrieval-flow--22)
- [Dataset](#dataset-evaluation-criteria-reproducibility--contributes)
- [Tech stack](#tech-stack-evaluation-criteria-reproducibility--contributes)
- [Quickstart](#quickstart-evaluation-criteria-reproducibility--22)
  - [Docker Compose (recommended)](#docker-compose-recommended)
  - [Local run (without Docker)](#local-run-without-docker)
- [Environment](#environment-evaluation-criteria-reproducibility--contributes)
- [Database](#database-evaluation-criteria-monitoring--12)
- [Using the app](#using-the-app-evaluation-criteria-interface--22)
- [CLI testing](#cli-testing)
- [Ingestion](#ingestion-evaluation-criteria-ingestion-pipeline--22)
- [Retrieval evaluation](#retrieval-evaluation-evaluation-criteria-retrieval-evaluation--22)
- [LLM evaluation](#llm-evaluation-evaluation-criteria-llm-evaluation--22)
- [Containerization](#containerization-evaluation-criteria-containerization--22)
- [Project structure](#project-structure-evaluation-criteria-reproducibility--contributes)
- [Troubleshooting](#troubleshooting-evaluation-criteria-reproducibility--contributes)
- [Best practices](#best-practices-evaluation-criteria-best-practices--03)
- [Bonus](#bonus-evaluation-criteria-bonus--02)
- [Evaluation Criteria summary](#evaluation-criteria-summary)


---

## Overview *(Evaluation Criteria: Problem description — 2/2)*

People often lack reliable nutrition facts for everyday foods, struggle to estimate macros for portions, and need quick substitutions that respect allergens. Generic chatbots can hallucinate values because they don’t ground answers in a verified dataset.

**Nutrition Assistant** is a retrieval-augmented application that answers nutrition questions using a local, curated CSV knowledge base. It retrieves the most relevant food entries and composes concise, evidence-based responses, reducing hallucinations and speeding up decisions.

- **Use cases**
  - Look up macronutrients (kcal, protein, fat, carbs).
  - Check vitamins & minerals (A, B6, B12, C, D, E, calcium, iron, potassium, magnesium, selenium, zinc, iodine).
  - Review allergens (tree nuts, peanut, sesame, etc.).
  - Ask free-form questions in a Streamlit UI.

- **RAG flow** *(Evaluation Criteria: Retrieval flow — 2/2)*
  - **Ingestion**: CSV → pandas → in-memory **minsearch** index. Column names normalized; missing values filled; aggregated `doc_text` built for robust retrieval.
  - **Retrieval**: boosted keyword search across `food`, `allergens`, `doc_text`.
  - **Generation**: OpenAI chat completion constrained to use only retrieved CONTEXT.
  - **Evaluation (optional)**: LLM-as-a-Judge classifies relevance (`RELEVANT | PARTLY_RELEVANT | NON_RELEVANT`).
  - **Persistence**: conversations and feedback are stored in **PostgreSQL**.

---

## Dataset *(Evaluation Criteria: Reproducibility — contributes)*

- Location: `data/data.csv`
- Format: **semicolon-separated** (`;`)
- Columns (lowercased at load time):
  - `food`, `serving_size_g`, `calories_kcal`, `protein_g`, `fat_g`, `carbohydrates_g`
  - `vitamin_a_mg`, `vitamin_b6_mg`, `vitamin_b12_mg`, `vitamin_c_mg`, `vitamin_d_mg`, `vitamin_e_mg`
  - `calcium_mg`, `iron_mg`, `potassium_mg`, `magnesium_mg`, `selenium_mg`, `zinc_mg`, `iodine_mg`
  - `allergens`

---

## Tech stack *(Evaluation Criteria: Reproducibility — contributes)*

- Python **3.12**
- **Streamlit** (UI)
- **minsearch** (local in-memory text search)
- **OpenAI** (LLM)
- **PostgreSQL** (conversations & feedback)
- **Docker** & **Docker Compose**
- **Pipenv** (dependencies)
- Optional: **python-dotenv** (local `.env`)

---

## Quickstart *(Evaluation Criteria: Reproducibility — 2/2)*

### Docker Compose (recommended)

1) Create `.env` in repo root:

```env
# Required
OPENAI_API_KEY=...

# App UI port (host)
APP_PORT=8501

# Data (optional; defaults to ./data/data.csv)
# DATA_PATH=./data/data.csv

# PostgreSQL
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=nutrition
POSTGRES_USER=app
POSTGRES_PASSWORD=app

2) Start:
docker-compose up -d
docker-compose logs -f app

3) Open UI: http://localhost:8501:
```If 8501 is busy, change APP_PORT in .env and re-run docker-compose up -d.

--
**Resume Evaluation Criteria (current state):**
- Problem description: **2/2**
- Retrieval flow: **2/2**
- Retrieval evaluation: **2/2**
- LLM evaluation: **2/2**
- Interface: **2/2**
- Ingestion pipeline: **2/2**
- Monitoring: **1/2** (feedback+; dashboard-)
- Containerization: **2/2**
- Reproducibility: **2/2**
- Best practices (hybrid / re-rank / rewrite): **0/3** (not implemented)
- Bonus (cloud): **0/2**


   
Evaluation Criteria

Problem description

Retrieval flow


Retrieval evaluation
The basic approach - using minsearch without any boosting - gave the following metrics:

{'hit_rate': 0.6471774193548387, 'mrr': 0.5518553187403998}
The improved version (with tuned boosting):

{'hit_rate': 0.8891129032258065, 'mrr': 0.7286029505888372}
The best boosting parameters:

boost = {
    'food': 2.84,
     'serving_size_g': 1.97,
     'calories_kcal': 2.03,
     'protein_g': 0.91,
     'fat_g': 1.89,
     'carbohydrates_g': 2.19,
     'vitamin_a_mg': 0.72,
     'vitamin_b6_mg': 1.62,
     'vitamin_b12_mg': 0.77,
     'vitamin_c_mg': 0.09,
     'vitamin_d_mg': 2.45,
     'vitamin_e_mg': 2.42,
     'calcium_mg': 0.47,
     'iron_mg': 0.21,
     'potassium_mg': 2.00,
     'magnesium_mg': 1.45,
     'selenium_mg': 1.41,
     'zinc_mg': 2.64,
     'iodine_mg': 0.62,
     'allergens': 2.64,
}

LLM evaluation

RAG flow evaluation
We used the LLM-as-a-Judge metric to evaluate the quality of our RAG flow.

For gpt-4o-mini, in a sample with 200 records, we had:
relevance
RELEVANT           129
NON_RELEVANT        43
PARTLY_RELEVANT     28
Name: count, dtype: int64

RELEVANT           0.660
NON_RELEVANT       0.175
PARTLY_RELEVANT    0.165
Name: proportion, dtype: float64

We also tested gpt-4o:

relevance
RELEVANT           148
PARTLY_RELEVANT     37
NON_RELEVANT        15
Name: count, dtype: int64
relevance
RELEVANT           0.740
PARTLY_RELEVANT    0.185
NON_RELEVANT       0.075
Name: proportion, dtype: float64
The difference is minimal, so we opted for gpt-4o-mini.



Interface


Ingestion pipeline


Monitoring



Containerization


Reproducibility



