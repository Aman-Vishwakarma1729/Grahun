import pandas as pd
from neo4j import GraphDatabase
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
import os
from src.reccommendation_duplicate import semantic_search
import streamlit as st
from input_to_search import preprocess_result

data_path = os.path.join(os.getcwd(), "dataset", "grahun_dataset_whole_1.csv")
data = pd.read_csv(data_path)
data = data[["Brand", "Product URL", "Product Price (USD)", "Currency", "Product ID", "Product Name", "Product Types", "Style", "Occasion", "Material", "Inspiration", "Trend"]]

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

similarity_threshold = 0.70

st.markdown("<h1 style='text-align: center;'>✨Grahun✨</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Recommendation System</h2>", unsafe_allow_html=True)
st.markdown("Find unique jewelry pieces based on your style and preferences.")

user_input = st.text_input("Describe your preference:")
if st.button("Submit"):
    stored_text = user_input
    output = preprocess_result(stored_text)
    for k,v in output.items():
        if k == "input_product_types":
           input_product_types = v
        elif k == "input_styles":
           input_styles = v
        elif k == "input_occasions":
             input_occasions = v
        elif k == "input_trends":
             input_trends = v
        elif k == "input_inspirations":
             input_inspirations = v
        elif k == "input_materials":
             input_materials = v

    with st.spinner("Aggregating products for you..."):
        with driver.session() as session:
            product_ids = session.read_transaction(
                semantic_search,
                input_product_types,
                input_materials,
                input_styles,
                input_occasions,
                input_trends,
                input_inspirations,
                similarity_threshold
            )

        filtered_df = data[data['Product ID'].isin(product_ids)]
        
        if len(filtered_df) > 0:
            st.markdown(f"### Total products found: {len(filtered_df)}")

            for _, row in filtered_df.iterrows():
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ccc; padding: 15px; border-radius: 10px; margin: 10px 0; background-color: #333333;">
                        <h3 style="color: #ff6347;">{row['Product Name']}</h3>
                        <p><strong>Brand:</strong> {row['Brand']}</p>
                        <p><strong>Price:</strong> {row['Product Price (USD)']} {row['Currency']}</p>
                        <a href="{row['Product URL']}" target="_blank" style="color: #4CAF50; text-decoration: none;">
                            🔗 View Product
                        </a>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            st.write("No products found based on the specified criteria.")
    
    driver.close()
