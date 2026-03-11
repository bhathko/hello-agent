import os
import sys
from typing import Optional, List, Dict

from google import genai
from google.genai import types

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_example import BaseExample
from dotenv import load_dotenv

load_dotenv()


class Example3Sample(BaseExample):
    """
    為本書 "Hello Agents" 定製的LLM客戶端。
    它使用 Google Gemini SDK 呼叫 Gemini 模型，並預設使用串流回應。
    """
    def __init__(self, model: str = None, apiKey: str = None):
        """
        初始化客戶端。優先使用傳入參數，如果未提供，則從環境變數載入。
        """
        self.model = model or os.getenv("MODEL_ID", "gemini-2.5-flash")
        apiKey = apiKey or os.getenv("GEMINI_API_KEY")

        if not all([self.model, apiKey]):
            raise ValueError("模型ID和API金鑰必須被提供或在.env檔案中定義。")

        self.client = genai.Client(api_key=apiKey)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        呼叫大語言模型進行思考，並回傳其回應。
        """
        print(f"🧠 正在呼叫 {self.model} 模型...")
        try:
            # 將 OpenAI 格式的訊息轉換為 Gemini 格式
            system_instruction = None
            gemini_contents = []

            for msg in messages:
                role = msg["role"]
                content = msg["content"]

                if role == "system":
                    system_instruction = content
                elif role == "user":
                    gemini_contents.append(types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=content)]
                    ))
                elif role == "model":
                    gemini_contents.append(types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=content)]
                    ))

            # 建構生成配置
            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction,
                response_mime_type="application/json"
            )

            # 呼叫 Gemini 串流 API
            response = self.client.models.generate_content_stream(
                model=self.model,
                contents=gemini_contents,
                config=config,
            )

            # 處理串流回應
            print("✅ 大語言模型回應成功:")
            collected_content = []
            for chunk in response:
                content = chunk.text or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在串流輸出結束後換行
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 呼叫LLM API時發生錯誤: {e}")
            return None

    def run(self):
        """Execute the sample logic for this example."""
        try:
            exampleMessages = [
                {"role": "system", "content": "You are a helpful assistant that writes Python code."},
                {"role": "user", "content": "寫一個冒泡排序演算法"}
            ]

            print("--- 呼叫LLM ---")
            responseText = self.think(exampleMessages)
            if responseText:
                print("\n\n--- 完整模型回應 ---")
                print(responseText)

        except ValueError as e:
            print(e)


# --- 客戶端使用範例 ---
if __name__ == '__main__':
    sample = Example3Sample()
    sample.run()
