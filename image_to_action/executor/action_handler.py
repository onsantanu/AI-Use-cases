import webbrowser
import subprocess

from search.duckduckgo_search import search_duckduckgo
from search.wikipedia_fallback import search_wikipedia_summary

# def execute_action(action_data: dict):
#     action = action_data.get("action")

#     if action == "search":
#         query = action_data.get("query", "")
#         url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
#         webbrowser.open(url)

#     elif action == "reminder":
#         task = action_data.get("task", "")
#         time = action_data.get("time", "")
#         _show_notification(f"Reminder: {task} at {time}")

#     elif action == "note":
#         print(f"📝 Note: {action_data.get('text')}")

def execute_action(action_data: dict):
    action = action_data.get("action")

    if action == "search":
        query = action_data.get("query", "")
        results = search_duckduckgo(query)

        if results:
            return {"type": "search_results", "query": query, "results": results}
        else:
            wiki = search_wikipedia_summary(query)
            if wiki:
                return {"type": "search_results", "query": query, "results": [wiki]}
            else:
                return {"type": "error", "message": f"No results found for: {query}"}

    elif action == "reminder":
        task = action_data.get("task", "")
        time = action_data.get("time", "")
        _show_notification(f"Reminder: {task} at {time}")
        return {"type": "info", "message": f"Reminder: {task} at {time}"}

    elif action == "note":
        return {"type": "note", "text": action_data.get("text")}


def _show_notification(message: str):
    subprocess.run(["notify-send", message])
