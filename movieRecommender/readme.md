 # 🎬 Movie Recommender (TMDB)

A simple Python tool that recommends movies from TMDB based on:

    Genre (e.g., Action, Comedy)

    Minimum rating (0–10)

    Mood → mapped to a genre (e.g., happy → Comedy, sad → Drama, thriller → Thriller)

    No filters? It falls back to Trending (Top 10 this week)

The script prints the title, rating, and a short plot for each recommendation.
✨ Features

    CLI prompts for genre, minimum rating, and mood

    Mood → Genre mapping (happy/sad/thriller)

    Graceful fallback to Trending when no filters are provided

    Top-10 results, sorted by popularity

🧰 Requirements

    Python 3.8+

    A free TMDB API key: https://www.themoviedb.org/

🚀 Setup
1) Create & activate a virtual environment (recommended)

Linux/macOS

python3 -m venv venv
source venv/bin/activate

Windows (PowerShell)

python -m venv venv
.\venv\Scripts\Activate.ps1

2) Install dependencies

pip install requests

3) Set your TMDB API key

Pick one:

    Environment variable (preferred)

export TMDB_API_KEY=YOUR_REAL_KEY_HERE        # Linux/macOS
setx TMDB_API_KEY "YOUR_REAL_KEY_HERE"        # Windows (new terminal after)

Inline: edit the script and replace:

    API_KEY = os.getenv("TMDB_API_KEY", "YOUR_TMDB_API_KEY")

    with your actual key (not recommended for public repos).

▶️ Run (CLI)

python movie_recommender.py

Then follow the prompts:

    Genre (press Enter to skip)

    Minimum rating (0–10, press Enter to skip)

    Mood: happy, sad, thriller (press Enter to skip)

Examples

🎥 Welcome to Movie Recommender
Enter a genre (or leave blank): Action
Minimum rating 0–10 (leave blank to skip): 7
Mood (happy/sad/thriller) or leave blank:

🎥 Welcome to Movie Recommender
Enter a genre (or leave blank):
Minimum rating 0–10 (leave blank to skip):
Mood (happy/sad/thriller) or leave blank: happy

Pressing Enter for everything → Trending Top-10 this week.
🙂 Mood → Genre Mapping

    happy → Comedy

    sad → Drama

    thriller → Thriller

🌐 (Optional) Run as a small API

If you also want to expose the same logic via HTTP (so other tools can call it), create a new file api.py:

# api.py
import os
from typing import Optional
from fastapi import FastAPI, Query
import uvicorn
import requests

API_KEY = os.getenv("TMDB_API_KEY", "YOUR_TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

MOOD_GENRES = {"happy": "Comedy", "sad": "Drama", "thriller": "Thriller"}

app = FastAPI(title="Movie Recommender API")

def get_genre_id(genre_name: str):
    r = requests.get(f"{BASE_URL}/genre/movie/list",
                     params={"api_key": API_KEY, "language": "en-US"},
                     timeout=10)
    r.raise_for_status()
    for g in r.json().get("genres", []):
        if g["name"].lower() == genre_name.lower():
            return g["id"]
    return None

def fetch_movies(genre: Optional[str], min_rating: Optional[float], mood: Optional[str]):
    # mood → genre
    if mood and mood.lower() in MOOD_GENRES:
        genre = MOOD_GENRES[mood.lower()]

    params = {
        "api_key": API_KEY,
        "language": "en-US",
        "sort_by": "popularity.desc",
        "page": 1
    }

    if genre:
        gid = get_genre_id(genre)
        if gid is not None:
            params["with_genres"] = gid

    # Trending if no filters
    if not genre and min_rating is None and not mood:
        r = requests.get(f"{BASE_URL}/trending/movie/week",
                         params={"api_key": API_KEY}, timeout=10)
    else:
        r = requests.get(f"{BASE_URL}/discover/movie", params=params, timeout=10)

    r.raise_for_status()
    movies = r.json().get("results", [])

    if min_rating is not None:
        movies = [m for m in movies if m.get("vote_average", 0) >= min_rating]

    # Top 10
    out = []
    for m in movies[:10]:
        out.append({
            "title": m.get("title") or m.get("name") or "Untitled",
            "rating": m.get("vote_average", 0.0),
            "overview": (m.get("overview") or "").strip(),
            "poster_path": m.get("poster_path"),
            "id": m.get("id")
        })
    return out

@app.get("/recommend")
def recommend(
    genre: Optional[str] = Query(None, description="TMDB genre name, e.g. 'Action'"),
    min_rating: Optional[float] = Query(None, ge=0, le=10, description="Minimum TMDB rating 0–10"),
    mood: Optional[str] = Query(None, description="happy/sad/thriller"),
):
    if API_KEY in ("", "YOUR_TMDB_API_KEY"):
        return {"error": "Set TMDB_API_KEY environment variable or edit API_KEY."}
    try:
        return {"results": fetch_movies(genre, min_rating, mood)}
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", 500)
        return {"error": f"HTTP {status}: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)

Install API deps & run:

pip install fastapi uvicorn requests
export TMDB_API_KEY=YOUR_REAL_KEY
python api.py

Call it:

# Trending Top 10
curl "http://127.0.0.1:8000/recommend"

# Filtered
curl "http://127.0.0.1:8000/recommend?genre=Action&min_rating=7.5"
curl "http://127.0.0.1:8000/recommend?mood=happy&min_rating=7"

You’ll get JSON like:

{
  "results": [
    {
      "title": "Example Movie",
      "rating": 7.6,
      "overview": "Short plot...",
      "poster_path": "/abc123.jpg",
      "id": 12345
    }
  ]
}

📁 Project Structure (suggested)

movie-recommender/
├─ README.md
├─ movie_recommender.py     # CLI script (your current code)
├─ api.py                   # optional API server (FastAPI)
├─ requirements.txt         # requests (and fastapi/uvicorn if using API)
└─ .env                     # optional, TMDB_API_KEY=...

requirements.txt (minimal CLI):

requests

If you’re using the API:

requests
fastapi
uvicorn

🧪 Troubleshooting

    Unauthorized / 401
    Make sure TMDB_API_KEY is set and valid.

    Nothing prints
    Ensure you actually run the function (the provided script calls main()).

    Rate limits / empty results
    TMDB may limit rate; try again or reduce frequency. Some filters can be too strict.

🔒 Notes

    Don’t commit your real API key to public repos. Use environment variables or .env (and add .env to .gitignore).

    Read TMDB Terms of Use & Attribution requirements.
