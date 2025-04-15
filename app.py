import os
import streamlit as st
from langchain.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv




# Retrieve API URL and API Key from environment variables
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Check if the API URL and API Key are available


# Setup the LLM with Hugging Face API
def get_mistral_llm(api_url: str, api_key: str):
    from langchain_community.llms import HuggingFaceEndpoint

    return HuggingFaceEndpoint(
        api_url=api_url,
        api_key=api_key,
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",  # Use appropriate model here
        temperature=0.7,
    )

llm = get_mistral_llm(API_URL, API_KEY)

# Setup memory to store the conversation
memory = ConversationBufferMemory(memory_key="chat_history", input_key="user_input")

# Setup prompt template for the recommendation
prompt_template = """
You are a friendly anime recommendation system. Based on the user's preferences, suggest an anime they might like.
User Input: {user_input}
"""
prompt = PromptTemplate(input_variables=["user_input"], template=prompt_template)
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Streamlit UI
st.title("Anime Recommendation Chatbot")
st.write("Tell me what kind of anime you like, and I will suggest a few based on your preferences!")

# User input
user_input = st.text_input("🎌 What kind of anime do you like?")

if user_input:
    # Get recommendation from the model
    response = llm_chain.run(user_input)
    
    # Display the recommendation
    st.write("🎬 Here is your anime recommendation:")
    st.write(response)
