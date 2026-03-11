import os
from typing import List
from openai import OpenAI

from app.core.retrieval import Evidence

_openai_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def build_evidence_block(evidence: List[Evidence]) -> str:
    blocks = []
    for i, e in enumerate(evidence[:6], 1):
        blocks.append(
            f"""[Evidence {i}]
doc_id: {e.doc_id}
vendor: {e.vendor}
version: {e.version}
pages: {e.page_start}-{e.page_end}
section: {e.section_title}
chunk_id: {e.chunk_id}
text:
{e.text}
"""
        )
    return "\n\n".join(blocks)


def answer_with_openai(question: str, evidence: List[Evidence]) -> str:
    client = get_openai_client()
    evidence_text = build_evidence_block(evidence)

    system_prompt = """
You are a hardware debugging assistant.
Answer ONLY from the provided evidence.
Do not use outside knowledge.
If the evidence is insufficient, say exactly:
"Insufficient evidence in the provided documents."
Always cite sources inline using:
[doc_id pages X-Y, chunk_id]
Keep the answer concise, technical, and audit-friendly.
"""

    response = client.responses.create(
        model="gpt-5.4",
        input=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""Question:
{question}

Evidence:
{evidence_text}

Instructions:
- Use only the evidence above.
- Do not infer beyond the evidence.
- Include citations inline.
"""
            }
        ],
    )

    return response.output_text.strip()