import os
import sys
from typing import Optional

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_chapter import BaseChapter
from .agent import Agent
from .config import get_config
from .llm_client import GeminiClient
from .tools import get_available_tools
from dotenv import load_dotenv

class Chapter2Sample(BaseChapter):
    """
    Chapter 2 Runner: Implementation of the ReAct Agent.
    """
    
    def run(self, user_prompt: Optional[str] = None):
        print("--- Running Chapter 2: Action-Thought-Observe Agent ---")
        
        # Load environment
        load_dotenv()
        config = get_config()

        # Validate
        is_valid, error_msg = config.validate()
        if not is_valid:
            print(f"Configuration Error: {error_msg}")
            return

        # Setup
        os.environ['TAVILY_API_KEY'] = config.tavily_api_key
        llm_client = GeminiClient(model=config.model_id, api_key=config.gemini_api_key)
        available_tools = get_available_tools()

        # Initialize and Run Agent
        agent = Agent(
            llm_client=llm_client,
            available_tools=available_tools,
            max_iterations=5
        )

        final_answer = agent.run(user_prompt)
        print("
" + "=" * 40)
        print(f"Final Result: {final_answer}")
        print("--- Chapter 2 Demo Completed ---")
