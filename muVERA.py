import faiss
import numpy as np
from encoder import Encoder
from scoring import muvera_score

class MuVERAIndex:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.doc_vectors = []
        self.doc_texts = []
        self.encoder = Encoder()

    def add_document(self, doc_text: str):
        D = self.encoder.encode_document(doc_text, dim=self.dim)
        d_summary = self.encoder.compress_document(D)
        self.index.add(d_summary.reshape(1, -1))
        self.doc_vectors.append(D)
        self.doc_texts.append(doc_text)

    def search(self, q_vector: np.ndarray, top_k: int = 3):
        D, I = self.index.search(q_vector.reshape(1, -1), top_k)
        return I[0]

    def rerank(self, query: str, top_k: int = 3):
        q = self.encoder.encode_query(query, dim=self.dim)
        initial_ids = self.search(q, top_k)
        results = []
        for idx in initial_ids:
            D = self.doc_vectors[idx]
            score = muvera_score(q, D)
            results.append((self.doc_texts[idx], score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
