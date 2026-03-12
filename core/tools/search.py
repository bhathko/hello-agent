import os

from serpapi import SerpApiClient


def search(query: str) -> str:
    """
    一個基於SerpApi的實戰網頁搜尋引擎工具。
    它會智慧地解析搜尋結果，優先返回直接答案或知識圖譜資訊。
    """
    print(f"🔍 正在執行 [SerpApi] 網頁搜尋: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "錯誤：SERPAPI_API_KEY 未在 .env 檔案中配置。"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "tw",  # 國家代碼
            "hl": "zh-tw", # 語言代碼
        }
        
        client = SerpApiClient(params)
        results = client.get_dict()
        
        # 智慧解析：優先尋找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果沒有直接答案，則返回前三個自然結果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"抱歉，沒有找到關於 '{query}' 的資訊。"

    except Exception as e:
        return f"搜尋時發生錯誤：{e}"
