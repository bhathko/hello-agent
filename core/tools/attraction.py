import os

from tavily import TavilyClient


def get_attraction(city: str, weather: str) -> str:
    """
    Search for attraction recommendations using the Tavily Search API based on city and weather.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable is not configured."

    tavily = TavilyClient(api_key=api_key)
    query = f"Recommended tourist attractions and reasons to visit in '{city}' during '{weather}' weather"

    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)

        if response.get("answer"):
            return response["answer"]

        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
            return "Sorry, no relevant tourist attraction recommendations were found."

        return "Based on the search, here is the information found:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"Error: A problem occurred while performing the Tavily search - {e}"
