# src/services/web_search_service.py
"""Web search service for fallback searches."""

from typing import List, Optional
from dataclasses import dataclass
import sys
from pathlib import Path
import requests

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

@dataclass
class WebSearchResult:
    """Web search result."""
    title: str
    url: str
    snippet: str

class WebSearchService:
    """Web search service using DuckDuckGo."""

    def __init__(self):
        """Initialize web search service."""
        print("Web Search Service initialized")

    def search(self, query: str) -> Optional[WebSearchResult]:
        """Search web and return best result."""
        try:
            print(f"Searching web for: {query}")
            
            # Use DuckDuckGo API
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            # Get the best result
            if data.get('Abstract'):
                result = WebSearchResult(
                    title=data.get('AbstractText', query),
                    url=data.get('AbstractURL', f"https://duckduckgo.com/?q={query.replace(' ', '+')}"),
                    snippet=data.get('Abstract', '')
                )
                print(f"Found web result: {result.title}")
                return result
            
            # Try related topics if no abstract
            for topic in data.get('RelatedTopics', []):
                if isinstance(topic, dict) and topic.get('Text'):
                    result = WebSearchResult(
                        title=topic.get('Text', '')[:100],
                        url=topic.get('FirstURL', f"https://duckduckgo.com/?q={query.replace(' ', '+')}"),
                        snippet=topic.get('Text', '')
                    )
                    print(f"Found web result: {result.title}")
                    return result
            
            # Fallback result
            return WebSearchResult(
                title=f"Web Search: {query}",
                url=f"https://duckduckgo.com/?q={query.replace(' ', '+')}",
                snippet=f"Web araması sonuçları: {query}"
            )
            
        except Exception as e:
            print(f"Web search error: {e}")
            # Return fallback even on error
            return WebSearchResult(
                title=f"Search: {query}",
                url=f"https://duckduckgo.com/?q={query.replace(' ', '+')}",
                snippet=f"Bu konu hakkında web araması yapabilirsiniz."
            )