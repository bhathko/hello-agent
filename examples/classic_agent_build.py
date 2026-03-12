from core import BaseExample, ClassicLLMAgent


class ClassicAgentExample(BaseExample):
    """
    Example 3: 展示如何使用 ClassicLLMAgent 呼叫大語言模型。
    """

    def run(self):
        """Execute the sample logic for this example."""
        try:
            agent = ClassicLLMAgent()
            exampleMessages = [
                {"role": "system", "content": "You are a helpful assistant that writes Python code."},
                {"role": "user", "content": "寫一個冒泡排序演算法"}
            ]

            print("--- 呼叫LLM ---")
            responseText = agent.think(exampleMessages)
            if responseText:
                print("\n\n--- 完整模型回應 ---")
                print(responseText)

        except ValueError as e:
            print(e)


def main():
    """Main execution point for Example 3."""
    sample = ClassicAgentExample()
    sample.run()


if __name__ == '__main__':
    main()
