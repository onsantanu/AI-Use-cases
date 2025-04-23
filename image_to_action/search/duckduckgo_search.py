from duckduckgo_search import DDGS

def search_duckduckgo(query: str, max_results=5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "text": r.get("body", ""),
                "link": r.get("href", "")
            })
    return results