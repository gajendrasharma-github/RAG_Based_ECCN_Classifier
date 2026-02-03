import google.generativeai as genai


class ECCNDecisionLLM:
    def __init__(self, api_key, model_name="models/gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def decide(self, product_description, retrieved_eccns):
        context_blocks = []
        for i, eccn in enumerate(retrieved_eccns, start=1):
            context_blocks.append(
                f"""
Candidate {i}:
{eccn['text']}
"""
            )

        context = "\n".join(context_blocks)

        prompt = f"""
You are an export control classification expert.

Your task:
- Choose the MOST appropriate ECCN code for the product.
- You MUST choose from the candidate ECCNs provided below.
- If none clearly match, respond with exactly: INSUFFICIENT_INFORMATION

Rules:
- Do NOT invent ECCN codes.
- Do NOT use outside knowledge.
- Base your decision ONLY on the provided ECCN descriptions.

Product Description:
{product_description}

Candidate ECCNs:
{context}

Output format (STRICT):
ECCN: <ecn_number or INSUFFICIENT_INFORMATION>
Reason: <1–3 concise sentences>
"""

        response = self.model.generate_content(prompt)
        return response.text.strip()
