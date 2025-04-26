import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory

# Function to initialize Flan LLM from HuggingFace Endpoint
def get_flan_llm(api_url, api_key):
    return HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V3-0324",
        temperature=0.7,
        huggingfacehub_api_token=api_key,
    )

# Function to create recommendation chain

def get_recommendation_chain(llm):
    # Create memory buffer to store conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="user_input")

    # Define a Flan-T5-friendly prompt template
    prompt = PromptTemplate(
        input_variables=["chat_history", "user_input", "anime_list"],
        template="""
You are an anime recommendation assistant.

Here is the conversation so far:
{chat_history}

The user said: "{user_input}"

Based on this, here are some suggested anime:
{anime_list}

Now, explain in a friendly, conversational way why these anime were recommended. 
Mention any themes, tone, or plot similarities. Make the user feel excited to watch them!
"""
    )

    # Create and return the LangChain with memory and the prompt
    return LLMChain(llm=llm, prompt=prompt, memory=memory)
