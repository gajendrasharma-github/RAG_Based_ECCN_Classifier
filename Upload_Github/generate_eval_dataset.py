import pandas as pd
import random
import time
import os
import google.generativeai as genai

# ----------------------------
# CONFIG
# ----------------------------
INPUT_CSV = "data/eccn.csv"
OUTPUT_CSV = "data/eval_dataset.csv"
SAMPLE_SIZE = 200
MODEL_NAME = "models/gemini-2.5-flash"
SLEEP_SECONDS = 0.5  # rate-limit safety


# INIT GEMINI

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(MODEL_NAME)


# PROMPT TEMPLATE
PROMPT_TEMPLATE = """
You are helping generate evaluation data.

Rewrite the following product description so that it:
- Is shorter and simpler
- Sounds like a real user or product listing
- Keeps the original meaning
- Does NOT mention regulations, ECCN codes, or compliance language

ONLY output the rewritten description.
Do NOT explain anything.

Original description:
"{description}"
"""

# LOAD & SAMPLE DATA

df = pd.read_csv(INPUT_CSV)

# Use only leaf nodes
df = df[df["is_leaf"] == True]

sampled_df = df.sample(n=SAMPLE_SIZE, random_state=42)

results = []

# GENERATE EVAL QUERIES

for idx, row in sampled_df.iterrows():
    original_desc = str(row["description_en"]).strip()

    # Skip empty descriptions
    if not original_desc or original_desc.lower() == "nan":
        continue

    prompt = PROMPT_TEMPLATE.format(description=original_desc)

    try:
        response = model.generate_content(prompt)
        rewritten = response.text.strip()

        results.append({
            "query_text": rewritten,
            "true_ecn": row["ecn_number"],
            "source_ecn_description": original_desc
        })

        print(f"Generated: {row['ecn_number']}")

    except Exception as e:
        print(f"Failed for {row['ecn_number']}: {e}")

    time.sleep(SLEEP_SECONDS)

# SAVE DATASET

eval_df = pd.DataFrame(results)
eval_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Evaluation dataset saved to: {OUTPUT_CSV}")
print(f"Total samples generated: {len(eval_df)}")
