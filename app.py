import streamlit as st
from memory_system import MemorySystem
from llm_client import OllamaLLM

st.set_page_config(page_title="Long-Form Memory Demo", layout="wide")

st.title("Long-Form Memory System (SQLite + FAISS + Ollama)")
st.caption("Chat assistant with retrieval memory. Sidebar includes reset + Ollama test.")

with st.sidebar:
    st.subheader("Controls")

    if "session_id" not in st.session_state:
        st.session_state.session_id = "demo_session"
    if "scope" not in st.session_state:
        st.session_state.scope = "project:hackathon"
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.session_id = st.text_input("session_id", st.session_state.session_id)
    st.session_state.scope = st.text_input("scope", st.session_state.scope)

    if "ms" not in st.session_state:
        st.session_state.ms = MemorySystem(db_path="memory.db", index_path="faiss.index")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset MemorySystem"):
            if "ms" in st.session_state:
                del st.session_state["ms"]
            st.rerun()
    with c2:
        if st.button("Clear Chat"):
            st.session_state.history = []
            st.rerun()

    st.divider()

    if st.button("Load Replay (from DB)"):
        ms = st.session_state.ms
        logs = ms.db.load_session(st.session_state.session_id)
        st.session_state.history = []
        for x in logs:
            st.session_state.history.append({"role": "user", "text": x["user_text"]})
            st.session_state.history.append({"role": "assistant", "text": x["assistant_text"]})
        st.success(f"Loaded {len(logs)} turns.")
        st.rerun()

    st.divider()
    st.subheader("Ollama")
    if st.button("Test Ollama"):
        try:
            llm = OllamaLLM(model="llama3.2:3b")
            out = llm.chat("You are a tester.", "Reply with exactly: OLLAMA_OK", timeout_s=30)
            st.success(out.strip())
        except Exception as e:
            st.error(f"Ollama not reachable: {e}")

    st.divider()
    st.subheader("Profile KV")
    st.json(st.session_state.ms.db.list_profile())

ms: MemorySystem = st.session_state.ms

left, right = st.columns([2, 1])

with right:
    st.subheader("This turn")
    st.write("Active memory + IDs will appear after you send a message.")

with left:
    st.subheader("Chat")

    # Render history
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["text"])

    user_text = st.chat_input("Type here...")
    if user_text:
        # Show user immediately
        st.session_state.history.append({"role": "user", "text": user_text})
        with st.chat_message("user"):
            st.write(user_text)

        # Compute response
        out = ms.handle_turn(
            session_id=st.session_state.session_id,
            user_text=user_text,
            scope=st.session_state.scope,
            topk=50,
        )

        assistant_text = out["assistant_text"]
        st.session_state.history.append({"role": "assistant", "text": assistant_text})

        # Show assistant immediately (fixes “second attempt” feel)
        with st.chat_message("assistant"):
            st.write(assistant_text)

        # Right panel
        with right:
            st.subheader("This turn")
            with st.expander("Active Memory Used (this turn)", expanded=True):
                st.write(out["memory_bundle"] if out["memory_bundle"] else "No memory retrieved.")
                st.write("Memory IDs used:", out["memory_ids_used"])

        # Keep UI consistent
        st.rerun()
