from core import BaseExample, ChatLLMClient, ToolExecutor, ReActAgent
from core.tools.search import search


class BasicAgentToolExample(BaseExample):
    """
    Example 4: 展示如何使用 ReActAgent 搭配工具（SerpApi）來回答問題。
    """

    def run(self):
        """Execute the sample logic for this example."""
        llm_client = ChatLLMClient()
        tool_executor = ToolExecutor()

        search_description = "一個網頁搜尋引擎。當你需要回答關於時事、事實以及在你的知識庫中找不到的資訊時，應使用此工具。"
        tool_executor.registerTool("Search", search_description, search)

        agent = ReActAgent(llm_client, tool_executor, max_steps=5)
        agent.run("NVIDIA最新的GPU型號是什麼")


def main():
    """Main execution point for Example 4."""
    sample = BasicAgentToolExample()
    sample.run()


if __name__ == '__main__':
    main()
