import os
from typing import List, Dict

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """
    簡單的 LLM 客戶端。
    使用非串流請求，適合單輪對話場景。
    """

    def __init__(self, model: str = None, api_key: str = None):
        """
        初始化 LLM 客戶端。

        Args:
            model: 模型 ID（例如 'gemini-2.5-flash'）
            api_key: API 金鑰
        """
        self.model_id = model or os.getenv("MODEL_ID", "gemini-2.5-flash")
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        呼叫 LLM 產生回應。

        Args:
            prompt: 使用者提示詞
            system_prompt: 系統提示詞

        Returns:
            產生的回應文字
        """
        print(f"🧠 正在呼叫 {self.model_id} 模型...")
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config={
                    'system_instruction': system_prompt
                }
            )
            answer = response.text
            print("✅ 模型回應成功。")
            return answer
        except Exception as e:
            print(f"❌ 呼叫 LLM API 時發生錯誤：{e}")
            return "錯誤：呼叫模型服務時發生問題。"


class ChatLLMClient:
    """
    支援多輪對話的串流 LLM 客戶端。
    接受 OpenAI 風格的訊息列表格式。
    """
    def __init__(self, model: str = None, api_key: str = None):
        """
        初始化客戶端。優先使用傳入參數，如果未提供，則從環境變數載入。
        """
        self.model = model or os.getenv("MODEL_ID", "gemini-2.5-flash")
        api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not all([self.model, api_key]):
            raise ValueError("模型 ID 和 API 金鑰必須被提供或在 .env 檔案中定義。")

        self.client = genai.Client(api_key=api_key)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        以多輪對話方式呼叫 LLM，使用串流回應。

        Args:
            messages: OpenAI 風格的訊息列表 [{"role": "system/user/model", "content": "..."}]
            temperature: 溫度參數（預設 0）

        Returns:
            完整的回應文字
        """
        print(f"🧠 正在呼叫 {self.model} 模型...")
        try:
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

            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction,
            )

            response = self.client.models.generate_content_stream(
                model=self.model,
                contents=gemini_contents,
                config=config,
            )

            print("✅ 大語言模型回應成功:")
            collected_content = []
            for chunk in response:
                content = chunk.text or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 呼叫 LLM API 時發生錯誤：{e}")
            return None
