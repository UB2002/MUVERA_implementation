from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load the model once at module load time
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModel.from_pretrained(MODEL_NAME)


class Encoder:

    @staticmethod
    def encode_text(text: str) -> np.ndarray:
        inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = _model(**inputs)

        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        
        return embedding.squeeze(0).cpu().numpy()

    @staticmethod
    def encode_document(text:str , numvector:int = 8 , dim:int = 128) ->   np.ndarray :
       
       sentences = text.split(".") 
       vectors = [Encoder.encode_text(s.strip()) for s in sentences if s.strip()]
       
       return np.stack(vectors)
    
    @staticmethod
    def encode_query(text:str, dim:int = 128) -> np.ndarray:
        return Encoder.encode_text(text)
    

    @staticmethod
    def compress_document(D: np.ndarray) -> np.ndarray:
        return np.mean(D, axis=0)

    


if __name__ == "__main__":
    doc1 = Encoder.encode_document("The quick brown fox jumps over the lazy dog. It was fast.")
    doc2 = Encoder.encode_document("Graph neural networks are powerful for structured data.")

    query = Encoder.encode_query("fast animals")

    print("Doc1 shape:", doc1.shape)  # (num_sentences, 384)
    print("Query shape:", query.shape)  # (384,)

    summary_vec = Encoder.compress_document(doc1)
    print("Compressed Doc1 shape:", summary_vec.shape)  # (384,)
