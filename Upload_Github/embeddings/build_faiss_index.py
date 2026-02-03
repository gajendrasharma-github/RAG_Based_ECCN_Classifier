from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

def build_faiss(documents):
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True)

    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, "eccn.index")

    with open("eccn_metadata.pkl", "wb") as f:
        pickle.dump(documents, f)


