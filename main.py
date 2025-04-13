# main.py
import os
api_key = os.getenv("HUGGINGFACE_API_KEY")
from llm_chain import get_mistral_llm, get_recommendation_chain

# Define the API URL and API key
TEMP_API_KEY = ""  # Replace with your actual Hugging Face API key
TEMP_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

# Initialize the LLM with the API URL and API key passed as parameters
llm = get_mistral_llm(api_url=TEMP_API_URL, api_key=TEMP_API_KEY)

# Create the recommendation chain with memory
chain = get_recommendation_chain(llm)

# Now you can use `chain` to process user input and generate anime recommendations
# Example (using chain to generate a recommendation):
# response = chain.run(user_input="What anime should I watch if I like action and adventure?")
user_input = "I like action-packed anime with strong female leads."

# Run the chain
response = chain.invoke({"input": user_input})

# Print the result
print("\n🤖 Recommended Anime:\n")
print(response["text"] if isinstance(response, dict) else response)
