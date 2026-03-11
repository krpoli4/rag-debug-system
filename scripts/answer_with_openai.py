import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = """
You are a hardware debugging assistant.
Answer ONLY from the provided evidence.
Do not use outside knowledge.
If the evidence is insufficient, say:
"Insufficient evidence in the provided documents."
Always cite sources using:
- doc_id
- page_start-page_end
- chunk_id
"""

def build_evidence_block(evidence: list[dict]) -> str:
    blocks = []
    for i, e in enumerate(evidence, 1):
        blocks.append(
            f"""[Evidence {i}]
doc_id: {e["doc_id"]}
pages: {e["page_start"]}-{e["page_end"]}
chunk_id: {e["chunk_id"]}
section: {e["section_title"]}
text:
{e["text"]}
"""
        )
    return "\n\n".join(blocks)

def answer_question(question: str, evidence: list[dict]) -> str:
    evidence_text = build_evidence_block(evidence)

    response = client.responses.create(
        model="gpt-5.4",
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""Question:
{question}

Evidence:
{evidence_text}

Instructions:
- Use only the evidence above.
- If evidence is insufficient, explicitly say so.
- Keep the answer concise and technical.
- Include citations inline in this format:
  [doc_id pages X-Y, chunk_id]
"""
            }
        ],
    )

    return response.output_text


if __name__ == "__main__":
    sample_evidence = [
        {
            "doc_id": "nvidia_gpu_debug_r560",
            "page_start": 13,
            "page_end": 13,
            "chunk_id": "nvidia_gpu_debug_r560::c00006",
            "section_title": "Running Field Diagnostics",
            "text": "Xid Description Action 31 Suspected Hardware Problems Contact the hardware vendor. They can run through their hardware diagnostic process."
        }
    ]

    question = "What does Xid 31 mean?"
    answer = answer_question(question, sample_evidence)
    print(answer)