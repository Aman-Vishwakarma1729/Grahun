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

def get_node_names(tx, label):
    query = f"MATCH (n:{label}) RETURN n.name AS name"
    result = tx.run(query)
    return [record["name"] for record in result]

def semantic_search(tx, input_product_types, input_materials, input_styles, input_occasions, input_trends, input_inspirations, threshold):
    product_types = get_node_names(tx, "Product_Type")
    materials = get_node_names(tx, "Material")
    styles = get_node_names(tx, "Style")
    occasions = get_node_names(tx, "Occasion")
    trends = get_node_names(tx, "Trend")
    inspirations = get_node_names(tx, "Inspiration")
    
    def find_matching_nodes(input_list, existing_list):
        if len(input_list) == 0:
            return None
        input_embeddings = get_embeddings(input_list)
        return {
            item for item in existing_list
            if any(calculate_cosine_similarity(get_embedding(item), input_embedding) >= threshold for input_embedding in input_embeddings)
        }

    categories = [
        (input_product_types, product_types, "Product_Type", "HAS_TYPE"),
        (input_materials, materials, "Material", "HAS_MATERIAL"),
        (input_styles, styles, "Style", "HAS_STYLE"),
        (input_occasions, occasions, "Occasion", "HAS_OCCASION"),
        (input_trends, trends, "Trend", "HAS_TREND"),
        (input_inspirations, inspirations, "Inspiration", "HAS_INSPIRATION")
    ]

    product_ids = None

    for input_list, existing_list, node_label, relationship in categories:
        matching_nodes = find_matching_nodes(input_list, existing_list)
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

            if (len(product_ids) <= 20):
                break

    return list(product_ids) if product_ids else "Please describe the what you want in more details"