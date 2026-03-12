import os

from core import BaseExample, LLMClient, ReActAgent, get_config
from core.prompts import build_react_prompt, ROLE_TRAVEL_ASSISTANT
from core.tools.weather import get_weather
from core.tools.attraction import get_attraction
from dotenv import load_dotenv


# 工具描述（用於提示詞）
TOOL_DESCRIPTIONS = """- `get_weather(city: str)`：呼叫 Google 天氣 API，取得指定城市的當前天氣（包括天氣狀況、溫度和濕度）。
- `get_attraction(city: str, weather: str)`：使用搜尋引擎，根據城市和天氣狀況推薦旅遊景點。"""


class ActionThoughtObserveExample(BaseExample):
    """
    Example 2: 展示 ReAct（推理 + 行動）模式的智慧體。
    使用 LLMClient 搭配天氣和景點推薦工具。
    """

    def run(self):
        """Execute the sample logic for this example."""
        print("--- Running Example 2: Action-Thought-Observe Agent ---")
        
        load_dotenv()
        config = get_config()

        is_valid, error_msg = config.validate()
        if not is_valid:
            print(f"Configuration Error: {error_msg}")
            return

        os.environ['TAVILY_API_KEY'] = config.tavily_api_key
        llm_client = LLMClient(model=config.model_id, api_key=config.gemini_api_key)

        available_tools = {
            'get_weather': get_weather,
            'get_attraction': get_attraction,
        }

        system_prompt = build_react_prompt(
            role_description=ROLE_TRAVEL_ASSISTANT,
            available_tools=TOOL_DESCRIPTIONS,
        )

        agent = ReActAgent(
            llm_client=llm_client,
            available_tools=available_tools,
            system_prompt=system_prompt,
            max_iterations=5
        )

        final_answer = agent.run()
        print("\n" + "=" * 40)
        print(f"Final Result: {final_answer}")
        print("--- Example 2 Demo Completed ---")


def main():
    """Main execution point for Example 2."""
    sample = ActionThoughtObserveExample()
    sample.run()


if __name__ == '__main__':
    main()
