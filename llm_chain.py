import os
import streamlit as st
from transformers import pipeline

# Fetch API key from environment (set via GitHub Secrets or manually for testing)
api_key = os.getenv("HUGGING_FACE_API_KEY")
api_url = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-V3-0324"  # Correct API URL

# Initialize the pipeline directly from Hugging Face
generator = pipeline("text-generation", model="deepseek-ai/DeepSeek-V3-0324", 
                     api_key=api_key)

# Streamlit UI
st.title("MAL-BOT")
st.write("Tell me what kind of anime you like, and I will suggest a few based on your preferences!")

# User input
user_input = st.text_input("🎌 What kind of anime do you like?")

if user_input:
    # Run the generator to get a response
    response = generator(user_input)
    
    # Display the response
    st.write("🎬 Here is your anime recommendation:")
    st.write(response[0]['generated_text'])
