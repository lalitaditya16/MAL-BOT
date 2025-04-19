# main.py
import os
api_key = os.getenv("HUGGINGFACE_API_KEY")
from llm_chain import get_flan_llm, get_recommendation_chain

# Define the API URL and API key
TEMP_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # Replace with your actual Hugging Face API key
TEMP_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

# Initialize the LLM with the API URL and API key passed as parameters
llm = get_flan_llm(api_url=TEMP_API_URL, api_key=TEMP_API_KEY)

# Create the recommendation chain with memory
chain = get_recommendation_chain(llm)

# Now you can use `chain` to process user input and generate anime recommendations
# Example (using chain to generate a recommendation):
# response = chain.run(user_input="What anime should I watch if I like action and adventure?")

