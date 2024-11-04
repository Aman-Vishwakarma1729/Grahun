import json
import numpy as np
from src.sentence_transformer import encode_sentence
import os

class EmbeddingAnalyzer:
    def __init__(self, dictionary):
        self.dictionary = dictionary
        
        self.style_embeddings_dict = self.load_dictionary(os.path.join(os.getcwd(),"artifacts","Style_embeddings.txt"))
        self.trend_embeddings_dict = self.load_dictionary(os.path.join(os.getcwd(),"artifacts","Trend_embeddings.txt"))
        self.material_embeddings_dict = self.load_dictionary(os.path.join(os.getcwd(),"artifacts","Material_embeddings.txt"))
        self.inspiration_embeddings_dict = self.load_dictionary(os.path.join(os.getcwd(),"artifacts","Inspiration_embeddings.txt"))
        self.occasion_embeddings_dict = self.load_dictionary(os.path.join(os.getcwd(),"artifacts","Occasion_embeddings.txt"))

    def load_dictionary(self, file_path):
        with open(file_path, 'r') as file:
            data = file.read()
            return json.loads(data)

    def cosi(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings."""
        dot_product = np.dot(emb1, emb2)
        norm_emb1 = np.linalg.norm(emb1)
        norm_emb2 = np.linalg.norm(emb2)
        if norm_emb1 == 0 or norm_emb2 == 0:
            return 0.0
        return dot_product / (norm_emb1 * norm_emb2)
   
    def get_high_similarities(self, ini, category_embeddings_dict, threshold=0.7, top_n=2):
        """Return a list of top N styles with cosine similarity greater than the threshold."""
        emb1 = encode_sentence(ini)
        similarity_scores = []

        for style, emb2 in category_embeddings_dict.items():
            avg_cos = self.cosi(emb1, emb2)
            if avg_cos > threshold:
                similarity_scores.append((style, avg_cos))

        # Sort styles by similarity score in descending order and select top N
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        top_similar_styles = [style for style, _ in similarity_scores[:top_n]]

        return top_similar_styles

    def analyze(self):
        results = {
            'Product Type': self.dictionary['Product Type'],
            'Style': self.get_high_similarities(self.dictionary['Style'], self.style_embeddings_dict),
            'Trend': self.get_high_similarities(self.dictionary['Trend'], self.trend_embeddings_dict),
            'Material': self.get_high_similarities(self.dictionary['Material'], self.material_embeddings_dict),
            'Inspiration': self.get_high_similarities(self.dictionary['Inspiration'], self.inspiration_embeddings_dict),
            'Occasion': self.get_high_similarities(self.dictionary['Occasion'], self.occasion_embeddings_dict)
        }
        return results
