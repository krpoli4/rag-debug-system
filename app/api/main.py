import traceback
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

from app.core.schemas import AskRequest, AskResponse, Citation, EvidenceView
from app.core.retrieval import retrieve, gate, get_qdrant_client, get_embed_model
from app.core.generation import answer_with_openai

app = FastAPI(
    title="RAG Debug System API",
    version="1.0.0",
    description="Domain-specific hardware debugging assistant for Dell PowerEdge and NVIDIA GPU manuals.",
)


@app.on_event("startup")
def startup_event():
    get_qdrant_client()
    get_embed_model()


@app.get("/")
def root():
    return {
        "message": "RAG Debug System API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        evidence = retrieve(req.question, limit=8)
        refuse, reason = gate(req.question, evidence)

        citations = [
            Citation(
                doc_id=e.doc_id,
                vendor=e.vendor,
                version=e.version,
                page_start=e.page_start,
                page_end=e.page_end,
                section_title=e.section_title,
                chunk_id=e.chunk_id,
                score=e.score,
            )
            for e in evidence[:6]
        ]

        evidence_views = [
            EvidenceView(
                doc_id=e.doc_id,
                vendor=e.vendor,
                version=e.version,
                page_start=e.page_start,
                page_end=e.page_end,
                section_title=e.section_title,
                chunk_id=e.chunk_id,
                score=e.score,
                preview=e.preview,
            )
            for e in evidence[:6]
        ]

        if refuse:
            return AskResponse(
                status="refuse",
                decision="refuse",
                reason=reason,
                suggestion="Provide the exact XID / error code / log line / server model, or add the missing vendor document.",
                citations=citations,
                evidence=evidence_views,
            )

        answer = answer_with_openai(req.question, evidence)

        return AskResponse(
            status="ok",
            decision="answer",
            answer=answer,
            citations=citations,
            evidence=evidence_views,
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "type": e.__class__.__name__,
            },
        )