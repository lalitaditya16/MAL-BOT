from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory

# Function to initialize Mistral LLM from HuggingFace Endpoint
def get_mistral_llm(api_url, api_key):
    return HuggingFaceEndpoint(  # You can use any compatible model here
        endpoint_url=api_url,
        huggingfacehub_api_token=api_key,
        temperature=0.7  # Control response creativity
    )

# Function to get LangChain recommendation chain with memory
def get_recommendation_chain(llm):
    # Create memory buffer to store conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="user_input")

    # Define the prompt template
    prompt = PromptTemplate(
        input_variables=["chat_history", "user_input", "anime_list"],
        template="""
The following is a conversation between a user and an anime recommendation assistant.
Conversation history:
{chat_history}

User input: "{user_input}"

Here are some anime recommendations based on the query:
{anime_list}

Give a warm, friendly explanation about why these anime were recommended.
"""
    )

    # Create and return the LangChain with memory and the prompt
    return LLMChain(llm=llm, prompt=prompt, memory=memory)
