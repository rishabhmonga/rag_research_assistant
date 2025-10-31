import streamlit as st
from rag_assistant import ask, set_topic, conversation_history

st.set_page_config(page_title="Local Multi-Turn RAG", layout="wide")
st.title("🧠 Local Multi-Turn RAG Research Assistant")

# Topic selector (persist across reruns)
topic = st.sidebar.text_input("Topic / Thread", value=st.session_state.get("topic", "default"))
if st.session_state.get("topic") != topic:
    from rag_assistant import conversation_history as hist  # just to force import binding
    set_topic(topic)
    st.session_state["topic"] = topic

engine = st.sidebar.selectbox("Search engine", ["auto","searxng","brave"], index=0)

if "history" not in st.session_state:
    # populate UI history from persisted file (already loaded into module var)
    st.session_state.history = list(conversation_history)

with st.form("chat"):
    q = st.text_input("Ask or continue the conversation:")
    submitted = st.form_submit_button("Send")

if submitted and q.strip():
    with st.spinner("Retrieving, reasoning & citing..."):
        ans = ask(q.strip(), engine=engine)
    st.session_state.history.append((q.strip(), ans))

# Render conversation
for hq, ha in st.session_state.history:
    st.markdown(f"**You:** {hq}")
    st.markdown(f"**Assistant:** {ha}")
    st.markdown("---")
