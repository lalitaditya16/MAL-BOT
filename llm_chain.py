import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

def get_flan_llm(api_key=None):
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key, temperature=0.7)

def get_recommendation_chain(llm):
    prompt = PromptTemplate(
        input_variables=["chat_history", "user_input", "anime_list"],
        template="""
        You are an anime recommendation assistant. Based on the user's preference and the list of anime below,
        suggest one or two anime that match their taste.

        Previous conversation:
        {chat_history}

        User likes:
        {user_input}

        Available anime:
        {anime_list}

        Your recommendation:
        """
    )
    memory = ConversationBufferMemory(memory_key="chat_history")
    return LLMChain(llm=llm, prompt=prompt, memory=memory)
