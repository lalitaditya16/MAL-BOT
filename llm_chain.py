import os
import streamlit as st
from transformers import pipeline

# Fetch API key from environment (set via GitHub Secrets or manually for testing)
api_key = st.secrets("OPEN_AI_API_KEY")

llm = ChatGroq(model="gemma2-9b-it", api_key=groq_api_key)
# Correct API URL

# Initialize the pipeline directly from Hugging Face


# Streamlit UI
st.title("MAL-BOT")
st.write("Tell me what kind of anime you like, and I will suggest a few based on your preferences!")

# User input
user_input = st.text_input("🎌 What kind of anime do you like?")

if user_input:
    # Run the generator to get a response
    response = generat(user_input)
    
    # Display the response
    st.write("🎬 Here is your anime recommendation:")
    st.write(response[0]['generated_text'])
