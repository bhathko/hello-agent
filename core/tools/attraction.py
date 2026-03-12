import os

from tavily import TavilyClient


def get_attraction(city: str, weather: str) -> str:
    """
    根據城市和天氣狀況，使用 Tavily 搜尋 API 搜尋並推薦旅遊景點。
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "錯誤：TAVILY_API_KEY 環境變數未設定。"

    tavily = TavilyClient(api_key=api_key)
    query = f"在 '{city}' 天氣為 '{weather}' 時推薦的旅遊景點及造訪理由"

    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)

        if response.get("answer"):
            return response["answer"]

        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
            return "抱歉，找不到相關的旅遊景點推薦。"

        return "根據搜尋結果，以下是找到的資訊：\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"錯誤：執行 Tavily 搜尋時發生問題 - {e}"
