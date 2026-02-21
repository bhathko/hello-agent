"""
Main entry point for the Hello-Agent application.
"""
import os
import sys

from dotenv import load_dotenv

from hello_agent.config import get_config
from hello_agent.llm_client import GeminiClient
from hello_agent.agent import Agent
from hello_agent.tools import get_available_tools


def main():
    """Main application entry point."""
    # Load configuration from .env file
    load_dotenv()
    
    config = get_config()

    # Validate configuration
    is_valid, error_msg = config.validate()
    if not is_valid:
        print(f"Configuration Error: {error_msg}")
        print("Please check your environment variables. Refer to the .env file.")
        sys.exit(1)

    # Set Tavily API key in environment for tools
    os.environ['TAVILY_API_KEY'] = config.tavily_api_key

    # Initialize Gemini client
    llm_client = GeminiClient(
        model=config.model_id,
        api_key=config.gemini_api_key
    )

    # Get available tools
    available_tools = get_available_tools()

    # Initialize agent
    agent = Agent(
        llm_client=llm_client,
        available_tools=available_tools,
        max_iterations=5
    )

    # Run agent with user prompt
    user_prompt = "Hello, please help me check the weather in London today, and then recommend a suitable tourist attraction based on the weather."
    final_answer = agent.run(user_prompt)

    print("\n" + "=" * 40)
    print(f"Final Result: {final_answer}")


if __name__ == "__main__":
    main()
