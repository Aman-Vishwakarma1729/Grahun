import re
from src.user_query_processor import get_fashion_advice
from src.embedding_generator import EmbeddingAnalyzer

def get_input_to_search(user_input):
    user_query = user_input
    advice = get_fashion_advice(user_query)
    analyzer = EmbeddingAnalyzer(advice)
    results = analyzer.analyze()
    return results

def preprocess_value(value):     
    value = value.replace("-", " ")            
    value = re.sub(r'[^A-Za-z0-9]', ' ', value)
    return value.lower().strip()

def preprocess_result(user_input):
    results = get_input_to_search(user_input)
    output = {}
    for k, v in results.items():
        if k == "Product Type":
            output["input_product_types"] = [preprocess_value(v)]
        else:
            v = [preprocess_value(value) for value in v]
            output[f"input_{k.lower()}s"] = list(set(v)) if v else []
    print(output)
    return output


if __name__ == "__main__":
    user_query = "I am a data scientist and i looking for budget friendly watch for daily use so that i can wear it in my office over formal cloths"
    preprocess_result(user_query)
