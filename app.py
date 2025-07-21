import os
import streamlit as st
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from llm_chain import get_flan_llm, get_recommendation_chain

# Fetch API key from environment (set via secrets or manually for testing)
api_key = st.sercrets("OPEN_API_KEY")




# Initialize the LLM
llm = ChatGroq(model="gemma2-9b-it", api_key=groq_api_key)


# Initialize the recommendation chain
chain = get_recommendation_chain(llm)

# Streamlit UI
st.title("MAL-BOT")
st.write("Tell me what kind of anime you like, and I will suggest a few based on your preferences!")

# User input
user_input = st.text_input("🎌 What kind of anime do you like?")

if user_input:
    # Example of setting chat history and anime list (can be dynamic)
    chat_history = "User started asking for anime recommendations."  # Can be a stored conversation history
    anime_list = "Naruto, Attack on Titan, One Piece"  # Placeholder for actual anime list

    # Run the recommendation chain with all required inputs
    inputs = {
        "chat_history": chat_history,
        "user_input": user_input,
        "anime_list": anime_list
    }
    response = chain.run(inputs)

    # Display the response
    st.write("🎬 Here is your anime recommendation:")
    st.write(response)
