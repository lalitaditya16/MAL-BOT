### mal_faiss.py
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

token='ad5fae79dc18b1456281b2f63875bdea'
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_anime_list(token, limit=500):
    url = "https://api.myanimelist.net/v2/anime/ranking"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "ranking_type": "all",
        "limit": limit,
        "fields": "title,synopsis,genres"
    }
    res = requests.get(url, headers=headers, params=params).json()
    return [entry["node"] for entry in res["data"]]

def build_text(anime):
    synopsis = anime.get("synopsis", "")
    genres = ', '.join([g['name'] for g in anime.get("genres", [])])
    return f"{anime['title']}: {synopsis}. Genres: {genres}"

def build_faiss_index(anime_list):
    texts = [build_text(anime) for anime in anime_list]
    embeddings = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, texts, anime_list

if __name__ == "__main__":
    print("📡 Fetching anime from MAL...")
    anime_list = get_anime_list()

    # Build FAISS index from the anime data
    index, texts, raw_anime_list = build_faiss_index(anime_list)

    print("✅ FAISS index built with anime data.")
