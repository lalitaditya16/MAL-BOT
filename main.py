import os
from llm_chain import get_flan_llm, get_recommendation_chain

# Fetch API key and URL from environment variables (assuming the API key is set as a secret)
api_key = os.getenv("HUGGINGFACE_API_KEY")
api_url = "https://api-inference.huggingface.co/models/google/flan-t5-base"

if not api_key:
    raise ValueError("API Key is missing! Please check your environment configuration.")

# Initialize the LLM with the API URL and API key passed as parameters
llm = get_flan_llm(api_url, api_key)

# Create the recommendation chain with memory
chain = get_recommendation_chain(llm)

# Now you can use `chain` to process user input and generate anime recommendations
# Example (using chain to generate a recommendation):
user_input = "What anime should I watch if I like action and adventure?"
response = chain.run(user_input=user_input)

print(response)

