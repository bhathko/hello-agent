"""
LLM client for making API calls to Gemini services using the modern google-genai SDK.
"""
from google import genai


class GeminiClient:
    """
    A client for calling Google's Gemini API using the modern SDK.
    """

    def __init__(self, model: str, api_key: str):
        """
        Initialize the Gemini client.

        Args:
            model: The model identifier to use (e.g., 'gemini-1.5-flash')
            api_key: API key for authentication
        """
        self.model_id = model
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """
        Call the Gemini API to generate a response.

        Args:
            prompt: The user prompt
            system_prompt: The system prompt

        Returns:
            The generated response text
        """
        print(f"Calling Gemini model ({self.model_id})...")
        try:
            # Using the modern SDK's preferred way to handle system instructions
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config={
                    'system_instruction': system_prompt
                }
            )
            # The modern SDK uses .text property
            answer = response.text
            print("Gemini model response successful.")
            return answer
        except Exception as e:
            print(f"Error occurred while calling Gemini API: {e}")
            return "Error: An error occurred while calling the Gemini model service."
