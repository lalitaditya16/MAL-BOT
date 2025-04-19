import os
import streamlit as st
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from llm_chain import get_flan_llm, get_recommendation_chain

# Load environment variables from .env file
load_dotenv()

# Fetch API key and URL from environment variables
api_key = os.getenv("HUGGINGFACE_API_KEY")
api_url = "https://api-inference.huggingface.co/models/google/flan-t5-base"

if not api_key:
    st.error("API Key not found! Please check your .env file.")
else:
    # Initialize the LLM with the API URL and API key passed as parameters
    llm = get_flan_llm(api_url, api_key)

    # Create the recommendation chain with memory
    chain = get_recommendation_chain(llm)

    # Streamlit UI
    st.title("MAL-BOT")
    st.write("Tell me what kind of anime you like, and I will suggest a few based on your preferences!")

    # User input
    user_input = st.text_input("🎌 What kind of anime do you like?")

    if user_input:
        # Get recommendation from the model
        response = chain.run(user_input=user_input)
        
        # Display the recommendation
        st.write("🎬 Here is your anime recommendation:")
        st.write(response)
