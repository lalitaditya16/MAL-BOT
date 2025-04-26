import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory

# Function to initialize Flan LLM from HuggingFace Endpoint
def get_flan_llm(api_url, api_key):
    return HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V3-0324",
        task='text-generation'
        temperature=0.7,
        huggingfacehub_api_token=api_key,
    )

# Function to create recommendation chain

def get_recommendation_chain(llm):
    # Create memory buffer to store conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="user_input")

    # Define a prompt template for generating anime recommendations
    prompt = PromptTemplate(
        input_variables=["user_input", "chat_history"],
        template="""
You are an anime recommendation assistant.

The user said: "{user_input}"

Based on this, recommend some anime titles that align with the user's preferences.

If you think some anime might be a great fit, mention why they would enjoy them based on the genres, themes, or tone.
"""
    )

    # Create and return the LangChain with memory and the prompt
    return LLMChain(llm=llm, prompt=prompt, memory=memory)
