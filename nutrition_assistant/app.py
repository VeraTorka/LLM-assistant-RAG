# app.py — Streamlit UI (English), saves answers & feedback to DB
import os
import uuid
import time
from typing import Any, Dict

import streamlit as st

# Важно: используем абсолютные импорты пакета
from nutrition_assistant.rag import rag            # rag(query) -> str | dict
from nutrition_assistant import db                 # db.save_conversation / db.save_feedback

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---------- utils ----------
def _normalize_answer_data(raw: Any, model_used: str, response_time_s: float) -> Dict[str, Any]:
    """
    Привести ответ rag() к полному словарю для db.save_conversation().
    Если rag() вернул строку — оборачиваем. Если словарь — дополняем недостающие поля.
    """
    if isinstance(raw, dict) and "answer" in raw:
        ad = dict(raw)  # copy
    else:
        ad = {"answer": "" if raw is None else str(raw)}

    ad.setdefault("model_used", model_used)
    ad.setdefault("response_time", float(response_time_s))
    ad.setdefault("relevance", "UNKNOWN")
    ad.setdefault("relevance_explanation", "")

    # токены/стоимость — по умолчанию 0, если rag() их не предоставил
    ad.setdefault("prompt_tokens", 0)
    ad.setdefault("completion_tokens", 0)
    ad.setdefault("total_tokens", ad["prompt_tokens"] + ad["completion_tokens"])

    ad.setdefault("eval_prompt_tokens", 0)
    ad.setdefault("eval_completion_tokens", 0)
    ad.setdefault("eval_total_tokens", ad["eval_prompt_tokens"] + ad["eval_completion_tokens"])

    ad.setdefault("openai_cost", 0.0)
    return ad


# ---------- page config ----------
st.set_page_config(page_title="Nutrition Assistant (RAG)", page_icon="🥗", layout="centered")

# ---------- session state ----------
if "history" not in st.session_state:
    # item = {id, question, answer, answer_data, ts}
    st.session_state.history = []

# ---------- UI ----------
st.title("🥗 Nutrition Assistant (RAG)")

with st.form("question_form", clear_on_submit=False):
    question = st.text_input("Question", placeholder="How much protein is in chicken breast?")
    submitted = st.form_submit_button("Ask")

if submitted:
    q = (question or "").strip()
    if not q:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                t0 = time.perf_counter()
                raw = rag(q)  # str | dict
                elapsed = time.perf_counter() - t0
            except Exception as e:
                st.error(f"RAG error: {e}")
                raw, elapsed = None, 0.0

        if raw:
            conv_id = str(uuid.uuid4())
            # нормализация полей для БД
            answer_data = _normalize_answer_data(raw, DEFAULT_MODEL, elapsed)
            answer_text = answer_data["answer"]

            # сначала пишем в БД
            db_err = None
            try:
                db.save_conversation(
                    conversation_id=conv_id,
                    question=q,
                    answer_data=answer_data,
                )
            except Exception as e:
                db_err = str(e)

            # сохраняем в историю для UI
            st.session_state.history.insert(0, {
                "id": conv_id,
                "question": q,
                "answer": answer_text,
                "answer_data": answer_data,
                "ts": int(time.time()),
                "db_error": db_err,
            })

            if db_err:
                st.error(f"Saved locally, but DB insert failed: {db_err}")
            else:
                st.success("Answer saved to database. See it below in History.")

st.subheader("History")
if not st.session_state.history:
    st.info("No conversations yet. Ask your first question above.")
else:
    for item in st.session_state.history:
        header = f"Q: {item['question']}"
        with st.expander(header):
            st.markdown("**Answer:**")
            st.write(item["answer"])

            if item.get("db_error"):
                st.warning(f"DB error for this item: {item['db_error']}")

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("👍 Helpful", key=f"up_{item['id']}"):
                    try:
                        db.save_feedback(conversation_id=item["id"], feedback=+1)
                        st.success("Thanks for your feedback!")
                    except Exception as e:
                        st.error(f"Failed to save feedback to DB: {e}")
            with c2:
                if st.button("👎 Not helpful", key=f"down_{item['id']}"):
                    try:
                        db.save_feedback(conversation_id=item["id"], feedback=-1)
                        st.info("Thanks, we recorded your feedback.")
                    except Exception as e:
                        st.error(f"Failed to save feedback to DB: {e}")
            with c3:
                st.caption(f"Conversation ID: {item['id']}")
