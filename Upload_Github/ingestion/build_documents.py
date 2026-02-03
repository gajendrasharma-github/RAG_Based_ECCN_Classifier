import pandas as pd

def build_documents(csv_path):
    df = pd.read_csv(csv_path)

    # ✅ KEEP ONLY LEAF NODES
    df = df[df["is_leaf"] == True]

    documents = []

    for _, row in df.iterrows():
        description = str(row.get("description_en", "")).strip()
        notes = str(row.get("notes", "")).strip()

        # Combine only semantic text
        text = f"{description}\n{notes}".strip()

        # Skip empty text
        if not text:
            continue

        documents.append({
            "ecn_number": row["ecn_number"],
            "text": text
        })

    return documents
