from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def encode_sentence(sentence):
    print(f"Input: {sentence}")
    
    embedding = model.encode(sentence, convert_to_tensor=False).tolist()
    
    return embedding

