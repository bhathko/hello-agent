import os
from tavily import TavilyClient


def get_attraction(city: str, weather: str) -> str:
    """
    Search for and return optimized attraction recommendations using the Tavily Search API based on city and weather.
    """
    # 1. Read the API key from environment variables
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable is not configured."

    # 2. Initialize the Tavily client
    tavily = TavilyClient(api_key=api_key)

    # 3. Construct a precise query
    query = f"Recommended tourist attractions and reasons to visit in '{city}' during '{weather}' weather"

    try:
        # 4. Call the API, include_answer=True returns a comprehensive summary
        response = tavily.search(query=query, search_depth="basic", include_answer=True)

        # 5. Tavily's results are clean and ready to use
        # response['answer'] is a summary answer based on all search results
        if response.get("answer"):
            return response["answer"]

        # If there's no summary answer, format the raw results
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
            return "Sorry, no relevant tourist attraction recommendations were found."

        return "Based on the search, here is the information found:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"Error: A problem occurred while performing the Tavily search - {e}"
