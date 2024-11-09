from openai import OpenAI
import os

class OpenAIWrapper:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def generate_image(self, prompt, size="1024x1024", quality="standard", model="dall-e-3"):
        """
        Generates an image using DALL-E.
        """
        try:
            response = self.client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                n=1,
            )
            return response.data[0].url
        except Exception as e:
            raise Exception(f"Error with OpenAI image generation: {e}")

    async def generate_text(self, prompt, model="gpt-4-turbo-preview", max_tokens=4000):
        """
        Generates text using GPT models.
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error with OpenAI text generation: {e}")