import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


class ECCNRetriever:
    def __init__(
        self,
        index_path="eccn.index",
        metadata_path="eccn_metadata.pkl",
        model_name="BAAI/bge-base-en-v1.5"
    ):
        self.index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        self.model = SentenceTransformer(model_name)

    def retrieve(self, query_text, top_k=5):
        query_vec = self.model.encode([query_text])
        query_vec = np.array(query_vec).astype("float32")

        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.metadata[idx])

        return results
