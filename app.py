import streamlit as st

import rag

st.set_page_config(page_title="Campus Knowledge OS", page_icon="🎓", layout="wide")

CATEGORIES = ["general 🧑‍🤝‍🧑", "academics 📚", "hostel 🏢", "exams 📚", "placements 👏", "fees 💴"]

# ---------- Sidebar: knowledge base management ----------
with st.sidebar:
    st.title("Campus Knowledge OS 👩‍🎓")
    st.caption("Ask questions about your college's documents. Runs fully offline with Ollama.")

    st.subheader("Add a document")
    category = st.selectbox("Category", CATEGORIES)
    files = st.file_uploader("PDF, TXT or MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Index documents", disabled=not files, type="primary"):
        for f in files:
            with st.spinner(f"Indexing {f.name}…"):
                try:
                    n = rag.ingest(f.name, f.getvalue(), category)
                    st.success(f"{f.name}: {n} passages added") if n else st.warning(f"{f.name}: no readable text")
                except Exception as e:
                    st.error(f"{f.name}: {e}. Is Ollama running and the embedding model pulled?")

    st.subheader("Knowledge base")
    sources = rag.list_sources()
    if not sources:
        st.caption("No documents yet.")
    for s in sources:
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**{s['source']}**  \n{s['category']} · {s['chunks']} passages")
        if c2.button("✕", key=f"del_{s['source']}", help="Remove document"):
            rag.delete_source(s["source"])
            st.rerun()

    st.subheader("Search scope")
    scope = st.selectbox("Category filter", ["All categories"] + CATEGORIES)

# ---------- Main: chat ----------
st.header("Ask the campus")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for p in msg["sources"]:
                    st.markdown(f"**[{p['n']}] {p['source']}**")
                    st.caption(p["text"][:300] + "…")

if not st.session_state.messages:
    st.info('Upload a handbook or circular, then ask something like "What is the minimum attendance for exams?"')

if question := st.chat_input("Ask a question…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            passages = rag.retrieve(question, None if scope == "All categories" else scope)
            if not passages:
                answer = "I couldn't find anything relevant. Upload a document or widen the category filter."
                st.markdown(answer)
            else:
                answer = st.write_stream(rag.stream_answer(question, passages))
                with st.expander("Sources"):
                    for p in passages:
                        st.markdown(f"**[{p['n']}] {p['source']}**")
                        st.caption(p["text"][:300] + "…")
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": passages})
        except Exception as e:
            st.error(f"Could not get an answer: {e}. Check that Ollama is running (`ollama serve`).")
