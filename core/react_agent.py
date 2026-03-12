"""
Agent execution logic for the ReAct (Reasoning + Acting) pattern.
"""
import re
from typing import List, Dict, Callable, Optional

from core.llm_client import LLMClient


class ReActAgent:
    """
    An agent that uses the ReAct (Reasoning + Acting) pattern to solve tasks.
    It parses LLM output for Thought/Action/Observation steps, executes tools,
    and iterates until a final answer is reached.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        available_tools: Dict[str, Callable],
        system_prompt: str = "",
        max_iterations: int = 5
    ):
        """
        Initialize the agent.

        Args:
            llm_client: The LLM client to use for generating responses
            available_tools: Dictionary mapping tool names to their functions
            system_prompt: The system prompt for the agent
            max_iterations: Maximum number of iterations to run
        """
        self.llm_client = llm_client
        self.available_tools = available_tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.prompt_history: List[str] = []

    def run(self, user_prompt: Optional[str] = None) -> str:
        """
        Run the agent to solve a user's request.

        Args:
            user_prompt: The user's request. If None, uses a default prompt.

        Returns:
            The final answer
        """
        if user_prompt is None:
            user_prompt = "Hello, please help me check the weather in Kaohsiung City today, and then recommend a suitable tourist attraction based on the weather."
            
        self.prompt_history = [f"User Request: {user_prompt}"]
        print(f"User Input: {user_prompt}\n" + "=" * 40)

        for i in range(self.max_iterations):
            print(f"--- Iteration {i + 1} ---\n")

            # Build prompt
            full_prompt = "\n".join(self.prompt_history)

            # Call LLM to think
            llm_output = self.llm_client.generate(full_prompt, system_prompt=self.system_prompt)

            # Truncate extra Thought-Action pairs
            llm_output = self._truncate_output(llm_output)
            print(f"Model Output:\n{llm_output}\n")
            self.prompt_history.append(llm_output)

            # Parse and execute action
            observation = self._execute_action(llm_output)

            # Check if task is finished
            if observation is None:
                # Task completed
                return self._extract_final_answer(llm_output)

            # Record observation
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "=" * 40)
            self.prompt_history.append(observation_str)

        return "Error: Maximum iterations reached, task incomplete."

    def _truncate_output(self, llm_output: str) -> str:
        """Truncate extra Thought-Action pairs from LLM output."""
        match = re.search(
            r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)',
            llm_output,
            re.DOTALL
        )
        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                print("Truncated redundant Thought-Action pairs")
                return truncated
        return llm_output

    def _execute_action(self, llm_output: str) -> Optional[str]:
        """
        Parse and execute the action from LLM output.

        Returns:
            Observation string, or None if task is finished
        """
        action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
        if not action_match:
            return "Error: Failed to parse Action field. Ensure your response strictly follows the 'Thought: ... Action: ...' format."

        action_str = action_match.group(1).strip()

        # Check if task is finished
        if action_str.startswith("Finish"):
            return None

        # Parse tool call
        tool_name_match = re.search(r"(\w+)\(", action_str)
        args_match = re.search(r"\((.*)\)", action_str)

        if not tool_name_match or not args_match:
            return f"Error: Could not parse tool call format '{action_str}'"

        tool_name = tool_name_match.group(1)
        args_str = args_match.group(1)
        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

        # Execute tool
        if tool_name in self.available_tools:
            try:
                observation = self.available_tools[tool_name](**kwargs)
                return observation
            except Exception as e:
                return f"Error: Error executing tool '{tool_name}' - {e}"
        else:
            return f"Error: Undefined tool '{tool_name}'"

    def _extract_final_answer(self, llm_output: str) -> str:
        """Extract the final answer from Finish action."""
        finish_match = re.match(r".*Finish\[(.*)\]", llm_output, re.DOTALL)
        if finish_match:
            final_answer = finish_match.group(1)
            print(f"Task completed, final answer: {final_answer}")
            return final_answer
        return "Error: Could not extract final answer"
