from anthropic import Anthropic
import os

class ClaudeWrapper:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

    async def generate_text(self, prompt):
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-latest",  # Changer le modèle
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Error with Claude API: {e}")