# nutrition_assistant/rag.py
import os
import json
from time import perf_counter
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

# важно: относительный импорт внутри пакета
from . import ingest

# --- env & client ---
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment")
client = OpenAI(api_key=API_KEY)

# --- index ---
index = ingest.load_index()

def search(query: str) -> List[Dict[str, Any]]:
    boost = {
        'food': 2.84, 'serving_size_g': 1.97, 'calories_kcal': 2.03, 'protein_g': 0.91,
        'fat_g': 1.89, 'carbohydrates_g': 2.19, 'vitamin_a_mg': 0.72, 'vitamin_b6_mg': 1.62,
        'vitamin_b12_mg': 0.77, 'vitamin_c_mg': 0.09, 'vitamin_d_mg': 2.45, 'vitamin_e_mg': 2.42,
        'calcium_mg': 0.47, 'iron_mg': 0.21, 'potassium_mg': 2.00, 'magnesium_mg': 1.45,
        'selenium_mg': 1.41, 'zinc_mg': 2.64, 'iodine_mg': 0.62, 'allergens': 2.64,
    }
    return index.search(query=query, filter_dict={}, boost_dict=boost, num_results=5)

PROMPT_TEMPLATE = """
You are a nutrition assistant. Answer the QUESTION based on the CONTEXT from the food database.
Use only the facts from the CONTEXT when answering the QUESTION. If the CONTEXT does not contain the answer, say you don't know.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

ENTRY_FIELDS = [
    "food","serving_size_g","calories_kcal","protein_g","fat_g","carbohydrates_g",
    "vitamin_a_mg","vitamin_b6_mg","vitamin_b12_mg","vitamin_c_mg","vitamin_d_mg",
    "vitamin_e_mg","calcium_mg","iron_mg","potassium_mg","magnesium_mg","selenium_mg",
    "zinc_mg","iodine_mg","allergens"
]

def _extract_doc(item: Dict[str, Any]) -> Dict[str, Any]:
    doc = item.get("document", item)
    # нормализуем ключи к lower()
    return {str(k).lower(): v for k, v in doc.items()}

def build_prompt(query: str, search_results: List[Dict[str, Any]], max_chars: int = 2000) -> str:
    blocks, used = [], 0
    for item in search_results:
        d = _extract_doc(item)
        lines = [f"{k}: {'' if d.get(k) is None else d.get(k)}" for k in ENTRY_FIELDS]
        block = "\n".join(lines) + "\n\n"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    context = "".join(blocks).strip()
    return PROMPT_TEMPLATE.format(question=query, context=context)

def llm(prompt: str, model: str = "gpt-4o-mini"):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = resp.choices[0].message.content.strip()
    usage = resp.usage or None
    # нормализуем usage к ожидаемым ключам
    tokens = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
    }
    return text, tokens

EVAL_PROMPT_TEMPLATE = """
You are an expert evaluator for a RAG system.
Classify the relevance of the Generated Answer to the Question as:
"NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".
Return strictly JSON (no code block).

Question: {question}
Generated Answer: {answer}

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[brief explanation]"
}}
""".strip()

def evaluate_relevance(question: str, answer: str):
    prompt = EVAL_PROMPT_TEMPLATE.format(question=question, answer=answer)
    evaluation_text, tokens = llm(prompt, model="gpt-4o-mini")
    try:
        parsed = json.loads(evaluation_text)
    except json.JSONDecodeError:
        parsed = {"Relevance": "UNKNOWN", "Explanation": "Failed to parse evaluation"}
    return parsed, tokens

def calculate_openai_cost(model: str, tokens: Dict[str, int]) -> float:
    if model == "gpt-4o-mini":
        # $0.15 / 1M input tok, $0.60 / 1M output tok (пример; скорректируйте при необходимости)
        return (tokens["prompt_tokens"] * 0.00015 + tokens["completion_tokens"] * 0.0006) / 1000.0
    return 0.0

def rag(query: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    t0 = perf_counter()
    results = search(query)
    prompt = build_prompt(query, results)
    answer_text, token_stats = llm(prompt, model=model)
    relevance_obj, rel_token_stats = evaluate_relevance(query, answer_text)
    took = perf_counter() - t0

    openai_cost = calculate_openai_cost(model, token_stats) + calculate_openai_cost(model, rel_token_stats)

    return {
        "answer": answer_text,
        "model_used": model,
        "response_time": took,
        "relevance": relevance_obj.get("Relevance", "UNKNOWN"),
        "relevance_explanation": relevance_obj.get("Explanation", ""),
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "total_tokens": token_stats["total_tokens"],
        "eval_prompt_tokens": rel_token_stats["prompt_tokens"],
        "eval_completion_tokens": rel_token_stats["completion_tokens"],
        "eval_total_tokens": rel_token_stats["total_tokens"],
        "openai_cost": openai_cost,
    }
