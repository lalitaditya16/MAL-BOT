import os
import streamlit as st
from llm_chain import get_flan_llm, get_recommendation_chain
from mal_faiss import get_anime_list, build_faiss_index, search_similar_anime

# Load keys
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
token = st.secrets.get("MAL_TOKEN", os.getenv("MAL_TOKEN"))

# Setup LLM and chain
llm = get_flan_llm(api_key)
chain = get_recommendation_chain(llm)

# Load and index anime
@st.cache_resource(show_spinner="Loading anime list and building index...")
def load_index():
    anime_list = get_anime_list(token)
    return build_faiss_index(anime_list)

index, texts, anime_data = load_index()

# Streamlit UI
st.title("🎌 MAL-BOT: Your Anime Recommender")
st.write("Tell me what kind of anime you like, and I’ll recommend something you'll love!")

user_input = st.text_input("🎭 What kind of anime do you like?")

if user_input:
    # Search similar anime using FAISS
    similar_anime = search_similar_anime(user_input, index, texts, anime_data, top_k=5)
    anime_titles = ', '.join([anime['title'] for anime in similar_anime])

    # Generate response using LLM
    inputs = {
        "chat_history": "User wants anime recommendations.",
        "user_input": user_input,
        "anime_list": anime_titles
    }
    response = chain.run(inputs)

    st.subheader("🎬 Recommended Anime:")
    st.write(response)
    st.markdown("---")
    st.markdown("**🔎 Top matches from MAL:**")
    for anime in similar_anime:
        st.markdown(f"**{anime['title']}**\n> {anime.get('synopsis', 'No description available.')}")
