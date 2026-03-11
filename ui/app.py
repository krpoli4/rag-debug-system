import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(
    page_title="Hardware Debug Assistant",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ Hardware Debug Assistant")
st.caption("Domain-specific troubleshooting assistant for NVIDIA GPUs and Dell PowerEdge servers")

st.markdown("### Try sample questions")
sample_questions = [
    "What does Xid 31 mean?",
    "How do I run DCGM diagnostics?",
    "What does 'No boot device found' mean on Dell PowerEdge?",
]

cols = st.columns(len(sample_questions))
for i, q in enumerate(sample_questions):
    if cols[i].button(q, key=f"sample_{i}"):
        st.session_state["question"] = q

question = st.text_input(
    "Ask a debugging question",
    value=st.session_state.get("question", ""),
    key="question_input"
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching manuals..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=60
                )
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")
            else:
                if response.status_code != 200:
                    st.error(f"API error: {response.status_code}")
                    st.text(response.text)
                else:
                    data = response.json()

                    if data["decision"] == "refuse":
                        st.error("❌ Not enough evidence in indexed manuals")
                        st.write(data["reason"])
                        if data.get("suggestion"):
                            st.info(data["suggestion"])
                    else:
                        st.success("✅ Answer")
                        st.markdown(data["answer"])

                    st.divider()
                    st.subheader("Sources")

                    for c in data.get("citations", []):
                        st.markdown(
                            f"""
**{c['vendor']} — {c['doc_id']}**

Section: {c['section_title']}  
Pages: {c['page_start']} - {c['page_end']}  
Chunk: `{c['chunk_id']}`  
Score: {round(c['score'], 3)}
"""
                        )
                        st.divider()

                    st.subheader("Retrieved Evidence")

                    for i, ev in enumerate(data.get("evidence", []), 1):
                        title = (
                            f"[{i}] {ev['vendor']} | {ev['doc_id']} | "
                            f"pages {ev['page_start']}-{ev['page_end']} | "
                            f"score {round(ev['score'], 3)}"
                        )
                        with st.expander(title):
                            st.markdown(f"**Section:** {ev['section_title']}")
                            st.markdown(f"**Chunk ID:** `{ev['chunk_id']}`")
                            st.markdown("**Preview:**")
                            st.code(ev["preview"][:300], language="text")