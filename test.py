# test.py — run a single RAG query locally (no HTTP)
import os
import argparse
import pandas as pd

# dotenv опционально
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# важно: пакетный импорт
from nutrition_assistant.rag import rag  # предполагается nutrition_assistant/__init__.py

def pick_question(gt_path: str) -> str:
    if os.path.exists(gt_path):
        # пробуем с запятой, если не вышло — с точкой с запятой
        for sep in (",", ";"):
            try:
                df = pd.read_csv(gt_path, sep=sep)
                if "question" in df.columns and len(df):
                    return df.sample(1).iloc[0]["question"]
            except Exception:
                continue
        raise RuntimeError(f"Could not read questions from {gt_path}")
    # fallback, если файла нет
    return "How much protein is in 100 g of almond butter?"

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--q", dest="question", help="Question to ask")
    parser.add_argument("--gt", dest="gt_path", default="./data/ground-truth-retrieval.csv",
                        help="Path to ground-truth CSV (default: ./data/ground-truth-retrieval.csv)")
    args = parser.parse_args()

    q = (args.question or "").strip()
    if not q:
        q = pick_question(args.gt_path)

    print("Question:", q)
    ans = rag(q)
    print("Answer:", ans)

if __name__ == "__main__":
    main()
