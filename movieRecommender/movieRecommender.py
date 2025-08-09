import os
import requests

# === CONFIGURATION ===
API_KEY = os.getenv("TMDB_API_KEY", "YOUR_TMDB_API_KEY")  # or paste your key here
BASE_URL = "https://api.themoviedb.org/3"

# === MOOD to Genre Mapping ===
MOOD_GENRES = {
    "happy": "Comedy",
    "sad": "Drama",
    "thriller": "Thriller"
}

def get_genre_id(genre_name: str):
    """Fetch genre ID from TMDB API given a genre name (case-insensitive)."""
    try:
        r = requests.get(f"{BASE_URL}/genre/movie/list",
                         params={"api_key": API_KEY, "language": "en-US"},
                         timeout=10)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[Error] Failed to fetch genre list: {e}")
        return None

    genres = r.json().get("genres", [])
    for g in genres:
        if g["name"].lower() == genre_name.lower():
            return g["id"]
    return None

def get_movies(genre=None, min_rating=None, mood=None):
    """Fetch and print movie recommendations based on filters."""
    # Mood → genre
    if mood and mood.lower() in MOOD_GENRES:
        genre = MOOD_GENRES[mood.lower()]

    params = {
        "api_key": API_KEY,
        "language": "en-US",
        "sort_by": "popularity.desc",
        "page": 1
    }

    # Add genre filter if present
    if genre:
        gid = get_genre_id(genre)
        if gid is not None:
            params["with_genres"] = gid
        else:
            print(f"[Info] Genre '{genre}' not found. Falling back to trending.")

    try:
        # No filters at all → trending
        if not genre and not min_rating and not mood:
            r = requests.get(f"{BASE_URL}/trending/movie/week",
                             params={"api_key": API_KEY}, timeout=10)
        else:
            r = requests.get(f"{BASE_URL}/discover/movie",
                             params=params, timeout=10)

        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if r.status_code == 401:
            print("[Error] Unauthorized. Check your TMDB API key.")
        else:
            print(f"[Error] HTTP error {r.status_code}: {e}")
        return
    except requests.exceptions.RequestException as e:
        print(f"[Error] Network error: {e}")
        return

    movies = r.json().get("results", [])

    # Apply rating filter locally if given
    if min_rating is not None:
        movies = [m for m in movies if m.get("vote_average", 0) >= min_rating]

    movies = movies[:10]

    if not movies:
        print("No movies found with the given filters.")
        return

    print("\n🎬 Recommended Movies:\n")
    for m in movies:
        title = m.get("title") or m.get("name") or "Untitled"
        rating = m.get("vote_average", 0)
        overview = (m.get("overview") or "No plot available.").strip()
        print(f"📌 {title} ({rating:.1f}/10)")
        print(f"   {overview}\n")

def main():
    # Ensure API key exists
    if API_KEY in ("", "YOUR_TMDB_API_KEY"):
        print("[Error] Please set your TMDB API key. Either:")
        print("  • Set environment variable TMDB_API_KEY, or")
        print("  • Replace YOUR_TMDB_API_KEY in the script.")
        return

    print("🎥 Welcome to Movie Recommender")
    genre = input("Enter a genre (or leave blank): ").strip() or None
    min_rating_str = input("Minimum rating 0–10 (leave blank to skip): ").strip()
    mood = input("Mood (happy/sad/thriller) or leave blank: ").strip() or None

    min_rating = None
    if min_rating_str:
        try:
            min_rating = float(min_rating_str)
        except ValueError:
            print("[Info] Invalid rating. Ignoring rating filter.")

    get_movies(genre=genre, min_rating=min_rating, mood=mood)

if __name__ == "__main__":
    main()

