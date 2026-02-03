from ingestion.build_documents import build_documents
from embeddings.build_faiss_index import build_faiss

CSV_PATH = "data/eccn.csv"

def main():
    print("🔧 Building ECCN documents...")
    documents = build_documents(CSV_PATH)
    print(f"✅ Documents prepared: {len(documents)}")

    print("\n🔧 Building FAISS index...")
    build_faiss(documents)

    print("\n✅ FAISS rebuild complete.")
    print("Artifacts created:")
    print(" - eccn.index")
    print(" - eccn_metadata.pkl")

if __name__ == "__main__":
    main()
