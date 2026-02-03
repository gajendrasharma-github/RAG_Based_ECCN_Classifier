from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from retriever.retrieve_candidates import ECCNRetriever
from llm.eccn_decision import ECCNDecisionLLM

# ----------------------------
# App Initialization
# ----------------------------
app = FastAPI(
    title="ECCN Classification API",
    description="RAG-based ECCN classifier using FAISS + Gemini",
    version="1.0.0"
)

# ----------------------------
# Load models ONCE (startup)
# ----------------------------
retriever = ECCNRetriever()
llm = ECCNDecisionLLM(
    api_key=os.getenv("GEMINI_API_KEY")
)

TOP_K = 5

# ----------------------------
# Request / Response Schemas
# ----------------------------
class ClassificationRequest(BaseModel):
    product_text: str


class ClassificationResponse(BaseModel):
    predicted_ecn: str
    reason: str
    retrieved_candidates: list[str]


# ----------------------------
# Health Check
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------------------
# Main Classification Endpoint
# ----------------------------
@app.post("/classify", response_model=ClassificationResponse)
def classify(request: ClassificationRequest):
    query = request.product_text.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Empty product text")

    # ---------- RETRIEVAL ----------
    retrieved = retriever.retrieve(query, top_k=TOP_K)
    retrieved_ecns = [r["ecn_number"] for r in retrieved]

    # Safety gate: no signal
    if not retrieved_ecns:
        return ClassificationResponse(
            predicted_ecn="INSUFFICIENT_INFORMATION",
            reason="No relevant ECCN candidates retrieved.",
            retrieved_candidates=[]
        )

    # ---------- LLM DECISION ----------
    llm_output = llm.decide(
        product_description=query,
        retrieved_eccns=retrieved
    )

    # Parse output
    predicted_ecn = "INSUFFICIENT_INFORMATION"
    reason = llm_output

    for line in llm_output.splitlines():
        if line.startswith("ECCN:"):
            predicted_ecn = line.replace("ECCN:", "").strip()
        if line.startswith("Reason:"):
            reason = line.replace("Reason:", "").strip()

    return ClassificationResponse(
        predicted_ecn=predicted_ecn,
        reason=reason,
        retrieved_candidates=retrieved_ecns
    )
