import os
import re
import logging
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer,util
from sklearn.metrics.pairwise import cosine_similarity
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
import streamlit as st

NEO4J_URI="neo4j+s://cc453036.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="lv3ZfIjWSG9owqU1MwVbxvRG0s8tdFiqGnU38NMAou4"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

data_path = os.path.join(os.getcwd(), "dataset", "grahun_dataset_whole_2.csv")
data = pd.read_csv(data_path)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Define a function to retrieve node names for any specified label
def get_child_span_names(driver, label):
    query = f"""
    MATCH (node:{label})
    RETURN node.name AS name
    """
    
    # Execute the query and return results
    with driver.session() as session:
        result = session.run(query)
        names = [record["name"] for record in result]
    return names
    
# Close the driver
driver.close()

def find_similar_words(user_words, label_words_dict, threshold=0.5):
    similar_words = {}

    # Encode user words
    user_embeddings = model.encode(user_words, convert_to_tensor=True)
    
    # Iterate over each label list
    for label, words in label_words_dict.items():
        # Encode label words
        label_embeddings = model.encode(words, convert_to_tensor=True)
        
        # Compute cosine similarities between user words and label words
        cosine_similarities = util.cos_sim(user_embeddings, label_embeddings)
        
        # Find words with similarity above threshold
        label_similar_words = []
        for i, user_word in enumerate(user_words):
            for j, label_word in enumerate(words):
                cosine_sim = cosine_similarities[i][j].item()
                if cosine_sim >= threshold:
                    label_similar_words.append((user_word, label_word, cosine_sim))
        
        if label_similar_words:
            similar_words[label] = label_similar_words
    
    return similar_words

def get_child_to_span_mapping(driver):
    query = """
    MATCH (child)
    WHERE child:Processed_Style_Child OR 
          child:Processed_Occasion_Child OR 
          child:Processed_Material_Child OR 
          child:Processed_Inspiration_Child OR 
          child:Processed_Trend_Child
    MATCH (child)-[:ALSO_REFERRED_AS]->(span)
    RETURN child.name AS child_name, collect(span.name) AS span_names
    """
    
    child_to_span_mapping = {}

    # Execute the query and build the mapping
    with driver.session() as session:
        result = session.run(query)
        for record in result:
            child_name = record["child_name"]
            span_names = record["span_names"]
            child_to_span_mapping[child_name] = span_names

    return child_to_span_mapping

def print_product_details(data, product_id):
    # Filter the DataFrame for the specific product ID
    product_details = data[data["Product ID"] == product_id]
    
    # Check if product ID exists in the DataFrame
    if not product_details.empty:
        # Extract details
        brand_name = product_details["Brand Name"].values[0]
        product_name = product_details["Product Name"].values[0]
        home_country = product_details["Home Country"].values[0]
        differentiation = product_details["Brand Differentiation/Uniqueness"].values[0]
        price_currency = f"{product_details['Product Price (USD)'].values[0]} {product_details['Currency'].values[0]}"
        product_description = product_details["Product Description"].values[0]
        
        # Use expander for collapsible product details
        with st.expander(f"Product Details for ID {product_id}"):
            st.write("Brand Name:", brand_name)
            st.write("Product Name:", product_name)
            st.write("Home Country:", home_country)
            st.write("Brand Differentiation/Uniqueness:", differentiation)
            st.write("Product Price and Currency:", price_currency)
            st.write("Product Description:", product_description)
    else:
        st.write("Product ID not found.")

def get_product_id_by_node_name(driver, node_name):
    query = """
    MATCH (relatedNode)
    WHERE relatedNode.name = $node_name
    MATCH (product:Product_IDs)-[:HAS_PRODUCT_TYPES|HAS_PROCESSED_STYLE|HAS_PROCESSED_OCCASION|HAS_PROCESSED_MATERIAL|HAS_PROCESSED_INSPIRATION|HAS_PROCESSED_TREND]->(relatedNode)
    RETURN product.name AS product_id
    """
    
    with driver.session() as session:
        result = session.run(query, node_name=node_name)
        product_ids = [record["product_id"] for record in result]
    
    return product_ids

# Define Streamlit UI
st.title("Recommendation System V-3")

# Input boxes for each category
material = st.text_input("Material", "").lower()
style = st.text_input("Style", "").lower()
occasion = st.text_input("Occasion", "").lower()
trend = st.text_input("Trend", "").lower()
inspiration = st.text_input("Inspiration", "").lower()

# Combine inputs into user_words list
user_words = [word for word in [material, style, occasion, trend, inspiration] if word]

# Threshold slider
st.write("Increasing the threshold lead to more number products in recommendation")
threshold = st.slider("Threshold", 0.0, 1.0, 0.6)

if st.button("Submit"):
    # Get names for each specific label
    inspiration_names = get_child_span_names(driver, "Inspiration_Child_Spans")
    style_names = get_child_span_names(driver, "Style_Child_Spans")
    occasion_names = get_child_span_names(driver, "Occasion_Child_Spans")
    material_names = get_child_span_names(driver, "Material_Child_Spans")
    trend_names = get_child_span_names(driver, "Trend_Child_Spans")

    # Now retrieving names for 'Processed' nodes
    processed_style_names = get_child_span_names(driver, "Processed_Style_Child")
    processed_occasion_names = get_child_span_names(driver, "Processed_Occasion_Child")
    processed_material_names = get_child_span_names(driver, "Processed_Material_Child")
    processed_inspiration_names = get_child_span_names(driver, "Processed_Inspiration_Child")
    processed_trend_names = get_child_span_names(driver, "Processed_Trend_Child")

    data = data

    st.write("Inspiration_Child_Spans Node Names:", len(inspiration_names))
    st.write("Style_Child_Spans Node Names:", len(style_names))
    st.write("Occasion_Child_Spans Node Names:", len(occasion_names))
    st.write("Material_Child_Spans Node Names:", len(material_names))
    st.write("Trend_Child_Spans Node Names:", len(trend_names))
    st.write("Total:",len(inspiration_names)+len(style_names)+len(occasion_names)+len(material_names)+len(trend_names))
    st.write(" ")
    st.write("Processed_Style_Child:", len(processed_style_names))
    st.write("Processed_Occasion_Child:", len(processed_occasion_names))
    st.write("Processed_Material_Child:", len(processed_material_names))
    st.write("Processed_Inspiration_Child:", len(processed_inspiration_names))
    st.write("Processed_Trend_Child:", len(processed_trend_names))
    st.write("Total:", len(processed_style_names)+len(processed_occasion_names)+len(processed_material_names)+len(processed_inspiration_names)+len(processed_trend_names))

    # Dictionary of all label words
    label_words_dict = {
        "Inspiration_Child_Spans": inspiration_names,
        "Style_Child_Spans": style_names,
        "Occasion_Child_Spans": occasion_names,
        "Material_Child_Spans": material_names,
        "Trend_Child_Spans": trend_names
    }

    # Create a dictionary of processed label words
    processed_label_words_dict = {
        "Inspiration_Child": processed_inspiration_names,
        "Style_Child": processed_style_names,
        "Occasion_Child": processed_occasion_names,
        "Material_Child": processed_material_names,
        "Trend_Child": processed_trend_names
    }

    # User input list and similarity threshold
    user_words = user_words
    threshold = threshold

    # Get similar words
    similar_words_spans = find_similar_words(user_words, label_words_dict, threshold)

    # Get the mapping and print it
    child_to_span_mapping = get_child_to_span_mapping(driver)

    # Print the results
    for label, words in similar_words_spans.items():
        st.subheader(f"Label: {label}")
        for user_word, label_word, cosine_val in words:
            st.write(f" - User Word: '{user_word}' matches Label Word: '{label_word}' with Cosine Similarity: {cosine_val}")
            for k,v in child_to_span_mapping.items():
                if label_word in v:
                    st.write(f" -- The child names is: {k}")
                    node_name = k
                    product_ids = get_product_id_by_node_name(driver, node_name)
                    st.write(f" ---- {len(product_ids)}")
                    for i in product_ids:
                        print_product_details(data, i)
            st.write(" ")
        st.write(f"Total matched words in {label}: {len(words)}")

    # Get similar words
    similar_words_child = find_similar_words(user_words, processed_label_words_dict, threshold)

    # Print the results
    for label, words in similar_words_child.items():
        st.subheader(f"Label: {label}")
        for user_word, matched_word, cosine_val in words:
            st.write(f" - User Word: '{user_word}' matches Processed Word: '{matched_word}' with Cosine Similarity: {cosine_val}")
            st.write(f" -- The child names is: {matched_word}")
            node_name = matched_word
            product_ids = get_product_id_by_node_name(driver, node_name)
            st.write(f" ---- {len(product_ids)}")
            for i in product_ids:
                print_product_details(data, i)
        st.write(" ")
        st.write(f"Total matched words in {label}: {len(words)}")