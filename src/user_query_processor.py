import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import ast

load_dotenv()

api_key = os.getenv("HUGGING_FACE_API_KEY")
client = InferenceClient(api_key=api_key)

def create_message(user_query):
    return [
        {"role": "system", "content": '''You are a fashion stylist. For the user's query, provide the output in the following format:\
                                         {"Style": "", "Occasion": "", "Trend": "", "Product Type": "", "Material": "", "Inspiration": ""}.\
                                         Use 'NA' for any category that is not applicable.'''}, 
        {"role": "user", "content": user_query},
    ]

def get_fashion_advice(user_query):
    message = create_message(user_query)

    outputs = client.chat_completion(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=message,
        max_tokens=512,
    )

    generated_content = outputs.choices[0].message['content']
    dictionary = ast.literal_eval(generated_content)
    return dictionary


if __name__ == "__main__":
     user_query = "I am a data scientist and i looking for budget friendly watch for daily use so that i can wear it in my office over formal cloths"
     dictionary = get_fashion_advice(user_query)
     print(dictionary)

