import re
import warnings
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return model.encode([text])[0]

def get_embeddings(text_list):
    return [get_embedding(text) for text in text_list]

def calculate_cosine_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

def preprocess_value(value):     
    value = value.replace("-", " ")            
    value = re.sub(r'[^A-Za-z0-9]', ' ', value)
    return value.lower().strip()

def get_node_names(tx, label):
    query = f"MATCH (n:{label}) RETURN n.name AS name"
    result = tx.run(query)
    return [record["name"] for record in result]

def semantic_search(tx, input_product_types, input_materials, input_styles, input_occasions, input_trends, input_inspirations, threshold):
    # Get available nodes for each category
    product_types = get_node_names(tx, "Product_Type")
    materials = get_node_names(tx, "Material")
    styles = get_node_names(tx, "Style")
    occasions = get_node_names(tx, "Occasion")
    trends = get_node_names(tx, "Trend")
    inspirations = get_node_names(tx, "Inspiration")

    # Semantic search only for input_product_types if non-empty
    if len(input_product_types) != 0:
        input_embeddings = get_embeddings(input_product_types)
        matching_product_types = {
            item for item in product_types
            if any(calculate_cosine_similarity(get_embedding(item), input_embedding) >= threshold for input_embedding in input_embeddings)
        }
    else:
        matching_product_types = set()

    # Process other categories if non-empty, otherwise skip
    def find_matching_nodes(input_list, existing_list):
        return {preprocess_value(item) for item in input_list if preprocess_value(item) in map(preprocess_value, existing_list)}

    categories = [
        (matching_product_types, "Product_Type", "HAS_TYPE"),
        (find_matching_nodes(input_materials, materials), "Material", "HAS_MATERIAL"),
        (find_matching_nodes(input_styles, styles), "Style", "HAS_STYLE"),
        (find_matching_nodes(input_occasions, occasions), "Occasion", "HAS_OCCASION"),
        (find_matching_nodes(input_trends, trends), "Trend", "HAS_TREND"),
        (find_matching_nodes(input_inspirations, inspirations), "Inspiration", "HAS_INSPIRATION")
    ]

    product_ids = None

    for matching_nodes, node_label, relationship in categories:
        if matching_nodes:
            query = (
                f"MATCH (p:Product_IDs)-[:{relationship}]->(n:{node_label}) "
                f"WHERE n.name IN $matching_nodes "
                f"RETURN p.product_id AS product_id"
            )
            result = tx.run(query, matching_nodes=list(matching_nodes))
            current_product_ids = {record["product_id"] for record in result}
            
            product_ids = current_product_ids if product_ids is None else product_ids.intersection(current_product_ids)
            
            print(f"After filtering by {node_label.lower()}: {len(product_ids)} Products")

            # If we have 10 or fewer products, break the loop early
            if len(product_ids) <= 10:
                break

    return list(product_ids) if product_ids else "Please describe in details"

