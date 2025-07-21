import os
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

token = os.getenv("MAL_TOKEN")
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_anime_list(token, limit=500):
    url = "https://api.myanimelist.net/v2/anime/ranking"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "ranking_type": "all",
        "limit": limit,
        "fields": "title,synopsis,genres"
    }
    res = requests.get(url, headers=headers, params=params)
    if res.status_code != 200:
        raise Exception(f"MAL API failed: {res.status_code} - {res.text}")
    return [entry["node"] for entry in res.json().get("data", [])]

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

def search_similar_anime(query, index, texts, anime_list, top_k=5):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return [anime_list[i] for i in indices[0]]
