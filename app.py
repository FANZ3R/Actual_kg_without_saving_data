"""
AxoraConnect Unified Knowledge API
=================================

Production-grade FastAPI interface over the UnifiedChatbot engine.

Responsibilities:
- Validate and normalize API inputs.
- Delegate all orchestration to UnifiedChatbot.
- Enforce a unified, stable response envelope.
- Provide pagination and metadata shaping.
- Expose health and query endpoints only.

Non-responsibilities:
- Retrieval logic
- LLM orchestration
- Vector DB / Knowledge Graph access
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from unified import UnifiedChatbot, ChatbotConfig

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("axoraconnect.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AxoraConnect Unified Knowledge API",
    version="1.0.0",
    description="Deterministic API for unified Vector DB + Knowledge Graph access",
)


# ---------------------------------------------------------------------------
# Unified Response Envelope
# ---------------------------------------------------------------------------

def build_response(
    code: int,
    success: bool,
    message: str,
    data: Optional[Any] = None,
    errors: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Standard response envelope for all AxoraConnect APIs.
    """
    return {
        "responseCode": code,
        "response": {
            "success": success,
            "message": message,
            "data": data,
            "errors": errors,
        },
    }


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """
    Query request for unified knowledge access.
    """

    query: str = Field(..., min_length=1, description="Natural language query")
    pageLength: int = Field(10, ge=1, le=50, description="Results per page")
    currentPage: int = Field(1, ge=1, description="Page number")


# ---------------------------------------------------------------------------
# Dependency Injection
# ---------------------------------------------------------------------------

def get_chatbot() -> UnifiedChatbot:
    """
    Initialize and provide the UnifiedChatbot engine.
    """
    try:
        config = ChatbotConfig()
        return UnifiedChatbot(config)
    except Exception as exc:
        logger.exception("Chatbot initialization failed")
        raise RuntimeError("Chatbot unavailable") from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["ops"])
def health_check():
    """Health probe for orchestration platforms."""
    return build_response(200, True, "Service healthy")


@app.post("/ask", tags=["query"])
def ask(
    request: AskRequest,
    chatbot: UnifiedChatbot = Depends(get_chatbot),
):
    """
    Unified knowledge query endpoint.

    Flow:
    1. Validate request
    2. Delegate to UnifiedChatbot
    3. Paginate and shape response
    4. Return unified envelope
    """
    try:
        result = chatbot.ask(request.query)
        metadata = result.get("metadata", {})

        items = result.get("items", [])

        # Pagination (API-level, deterministic)
        start = (request.currentPage - 1) * request.pageLength
        end = start + request.pageLength
        paginated = items[start:end]

        return build_response(
            code=200,
            success=True,
            message="Query successful",
            data={
                "question": result.get("question"),
                "answer": result.get("answer"),
                "results": paginated,
                "pagination": {
                    "currentPage": request.currentPage,
                    "pageLength": request.pageLength,
                    "totalResults": len(items),
                },
                "metrics": {
                    "vector_count": metadata.get("vector_count", 0),
                    "kg_count": metadata.get("kg_count", 0),
                    "search_time": metadata.get("search_time", 0.0),
                    "answer_time": metadata.get("answer_time", 0.0),
                    "total_time": metadata.get("total_time", 0.0),
                },
            },
        )

    except HTTPException as he:
        logger.warning(f"Request error: {he.detail}")
        return build_response(
            code=he.status_code,
            success=False,
            message="Invalid request",
            errors={"detail": he.detail},
        )

    except Exception as exc:
        logger.exception("Unhandled query failure")
        return build_response(
            code=500,
            success=False,
            message="Internal server error",
            errors={"detail": str(exc)},
        )


# ---------------------------------------------------------------------------
# Local Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Local development runner.

    HOST=0.0.0.0 PORT=8080 python app.py
    """
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8080")),
        reload=True,
    )