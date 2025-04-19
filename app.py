import os
import streamlit as st
from langchain.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from llm_chain import get_flan_llm, get_recommendation_chain



# Retrieve API URL and API Key from environment variables
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Check if the API URL and API Key are available


# Setup the LLM with Hugging Face APi
def get_flan_llm(api_url: str, api_key: str):
    return HuggingFaceEndpoint(
        endpoint_url=api_url,
        huggingfacehub_api_token=api_key,
        task="text2text-generation"
    )

llm = get_flan_llm(API_URL, API_KEY)

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
st.title("MAL-BOT")
st.write("Tell me what kind of anime you like, and I will suggest a few based on your preferences!")

# User input
user_input = st.text_input("🎌 What kind of anime do you like?")

if user_input:
    # Get recommendation from the model
    response = llm_chain.run(user_input)
    
    # Display the recommendation
    st.write("🎬 Here is your anime recommendation:")
    st.write(response)
