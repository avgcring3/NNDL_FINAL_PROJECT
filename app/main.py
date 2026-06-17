from __future__ import annotations

import streamlit as st

from app.config import DEMO_QUESTIONS
from app.generator import generate_answer
from app.retriever import index_exists, load_index, retrieve


st.set_page_config(page_title="RideFlow RAG", page_icon="RF", layout="wide")
st.title("RideFlow RAG")

if not index_exists():
    st.error("Index is not built yet. Run: python scripts/build_index.py")
    st.stop()

rag_index = load_index()

with st.sidebar:
    st.header("Query")
    selected_demo = st.selectbox("Demo question", DEMO_QUESTIONS)
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)

question = st.text_input("Question", value=selected_demo)

if question:
    results = retrieve(question, index=rag_index, top_k=top_k)
    generated = generate_answer(question, results)

    st.subheader("Answer")
    st.markdown(generated["answer"])

    st.subheader("Sources")
    if generated["sources"]:
        for source in generated["sources"]:
            with st.expander(
                f"{source['doc_id']} | {source['name']} | score={source['score']:.3f}"
            ):
                st.write(source["text"])
    else:
        st.info("No relevant source passed the relevance threshold.")

    st.subheader("Retrieved Fragments")
    for item in results:
        with st.expander(
            f"#{item['rank']} {item['doc_id']} | score={item['score']:.3f} | {item['name']}"
        ):
            st.write(item["text"])
