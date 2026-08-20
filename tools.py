import os
import requests
from tavily import TavilyClient
from bs4 import BeautifulSoup
from dotenv import load_dotenv 
from langchain.tools import tool
from rich import print
load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query:str) -> str:
#     ↓ This doc string is important to write because, it tells LLM the use case of this particular tool.
    """  Search the web for recent and reliable information on a topic. Returns Titles, URLs and snippets. """
    result_from_the_web = tavily.search(query = query, max_results=2)
    out = []

    for r in result_from_the_web['results']:
        out.append(
            f"Title: {r['title']}\nURL: {r['url']}\nSnippet:{r['content'][:300]}\n"
        )
    return "\n---\n".join(out)


@tool
def scrape_url(url: str)-> str:
    """ Scrape and return clean text content from a given URL for deeper reading. """
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"


















