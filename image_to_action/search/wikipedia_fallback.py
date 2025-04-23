import requests

def search_wikipedia_summary(query):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return {
            "title": data.get("title"),
            "text": data.get("extract"),
            "link": data.get("content_urls", {}).get("desktop", {}).get("page")
        }

    return None
