from tavily import TavilyClient
import os

class TavilyWrapper:
    def __init__(self):
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    async def search(self, query, search_depth="basic", max_results=5):
        """
        Performs an internet search using Tavily API.
        """
        try:
            response = self.client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results
            )
            return response
        except Exception as e:
            raise Exception(f"Error with Tavily search: {e}")