import pandas as pd
import os
import re
from retriever.retrieve_candidates import ECCNRetriever
from llm.eccn_decision import ECCNDecisionLLM

# ----------------------------
# CONFIG
# ----------------------------
EVAL_CSV = "data/eval_dataset.csv"
OUTPUT_CSV = "data/eval_results.csv"
TOP_K = 5

# ----------------------------
# INIT COMPONENTS
# ----------------------------
retriever = ECCNRetriever() if hasattr(ECCNRetriever, "__init__") else ECCNRetriever()
llm = ECCNDecisionLLM(api_key=os.getenv("GEMINI_API_KEY"))

# ----------------------------
# HELPERS
# ----------------------------
def extract_parent(ecn):
    """
    Extract parent ECCN.
    Example:
    0A504.B -> 0A504
    3A001.a -> 3A001
    """
    if ecn is None:
        return None
    return ecn.split(".")[0]

def parse_prediction(llm_output):
    """
    Extract predicted ECCN from LLM output.
    """
    match = re.search(r"ECCN:\s*([A-Z0-9\.]+|INSUFFICIENT_INFORMATION)", llm_output)
    if match:
        return match.group(1)
    return "PARSE_ERROR"

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(EVAL_CSV)

results = []

# ----------------------------
# EVALUATION LOOP
# ----------------------------
for idx, row in df.iterrows():
    query = row["query_text"]
    true_ecn = row["true_ecn"]

    # ---------- RETRIEVAL ----------
    retrieved = retriever.retrieve(query, top_k=TOP_K)
    retrieved_ecns = [r["ecn_number"] for r in retrieved]

    recall_at_k = true_ecn in retrieved_ecns

    # ---------- LLM DECISION ----------
    llm_output = llm.decide(
        product_description=query,
        retrieved_eccns=retrieved
    )

    predicted_ecn = parse_prediction(llm_output)

    # ---------- METRICS ----------
    exact_match = predicted_ecn == true_ecn
    parent_match = (
        extract_parent(predicted_ecn) == extract_parent(true_ecn)
        if predicted_ecn not in ["INSUFFICIENT_INFORMATION", "PARSE_ERROR"]
        else False
    )

    abstained = predicted_ecn == "INSUFFICIENT_INFORMATION"

    results.append({
        "query_text": query,
        "true_ecn": true_ecn,
        "predicted_ecn": predicted_ecn,
        "exact_match": exact_match,
        "parent_match": parent_match,
        "recall_at_k": recall_at_k,
        "abstained": abstained,
        "llm_output": llm_output
    })

    print(f"[{idx+1}/{len(df)}] Done")

# ----------------------------
# SAVE RESULTS
# ----------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)

# ----------------------------
# AGGREGATE METRICS
# ----------------------------
total = len(results_df)

exact_acc = results_df["exact_match"].mean()
parent_acc = results_df["parent_match"].mean()
recall_k = results_df["recall_at_k"].mean()
abstain_rate = results_df["abstained"].mean()

print("\n===== EVALUATION SUMMARY =====")
print(f"Samples evaluated        : {total}")
print(f"Exact Match Accuracy     : {exact_acc:.3f}")
print(f"Parent Match Accuracy    : {parent_acc:.3f}")
print(f"Recall@{TOP_K}           : {recall_k:.3f}")
print(f"Abstention Rate          : {abstain_rate:.3f}")
print(f"Results saved to         : {OUTPUT_CSV}")

