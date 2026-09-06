from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from ddgs import DDGS


class WebSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="The search query to find current information on the web."
    )


class WebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = (
        "Searches the web for current information and returns relevant "
        "search results with titles, links, and snippets."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        try:
            results = DDGS().text(
                query,
                max_results=5
            )

            if not results:
                return "No relevant web search results were found."

            output = []

            for i, result in enumerate(results, start=1):
                title = result.get("title", "No title")
                url = result.get("href", "No URL")
                snippet = result.get("body", "No description")

                output.append(
                    f"{i}. {title}\n"
                    f"URL: {url}\n"
                    f"Summary: {snippet}"
                )

            return "\n\n".join(output)

        except Exception as e:
            return f"Web search error: {str(e)}"