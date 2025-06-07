# src/services/web_search_service.py
"""Simple web search service using DuckDuckGo search."""

from typing import Optional
from dataclasses import dataclass
import sys
from pathlib import Path
import requests
from urllib.parse import quote
import re

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

@dataclass
class WebSearchResult:
    """Web search result."""
    title: str
    url: str
    snippet: str

class WebSearchService:
    """Simple web search service using DuckDuckGo lite."""

    def __init__(self):
        """Initialize web search service."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        print("Web Search Service initialized")

    def search(self, query: str) -> Optional[WebSearchResult]:
        """Search web and return the best result with title, link, and snippet."""
        try:
            print(f"Searching web for: {query}")
            
            # Search using DuckDuckGo lite
            result = self._search_ddg_lite(query)
            
            if result:
                print(f"Found result: {result.title}")
                return result
            
            # If no results, return None instead of fallback
            print("No search results found")
            return None
            
        except Exception as e:
            print(f"Web search error: {e}")
            return None

    def _search_ddg_lite(self, query: str) -> Optional[WebSearchResult]:
        """Search using DuckDuckGo lite HTML interface."""
        try:
            # Use DuckDuckGo lite interface
            url = "https://lite.duckduckgo.com/lite/"
            params = {'q': query}
            
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                html = response.text
                
                # Find the first good result
                # Pattern for DuckDuckGo lite: look for table rows with links
                pattern = r'<td[^>]*><a[^>]*href="(https?://[^"]+)"[^>]*>([^<]+)</a></td>\s*<td[^>]*>([^<]+)</td>'
                matches = re.findall(pattern, html, re.DOTALL)
                
                # Get the first valid result
                for url_found, title, snippet in matches:
                    # Clean up the data
                    title = self._clean_text(title)
                    snippet = self._clean_text(snippet)
                    url_found = url_found.strip()
                    
                    # Validate the result
                    if (len(title) > 10 and 
                        len(snippet) > 20 and 
                        self._is_valid_url(url_found)):
                        
                        return WebSearchResult(
                            title=title,
                            url=url_found,
                            snippet=snippet[:200] + "..." if len(snippet) > 200 else snippet
                        )
            
            return None
            
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove extra characters
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        
        return text.strip()
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and not an internal link."""
        if not url or not url.startswith('http'):
            return False
        
        # Skip unwanted domains
        unwanted = ['duckduckgo.com', 'google.com/search', 'bing.com/search']
        for domain in unwanted:
            if domain in url.lower():
                return False
        
        return len(url) > 15