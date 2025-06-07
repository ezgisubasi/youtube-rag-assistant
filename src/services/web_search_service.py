# src/services/web_search_service.py
"""Web search service using DuckDuckGo search - returns only the top result."""

from typing import Optional
from dataclasses import dataclass
import sys
from pathlib import Path
import requests
from urllib.parse import quote, unquote
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
    """Simple web search service - returns only the top result."""

    def __init__(self):
        """Initialize web search service."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        print("Web Search Service initialized")

    def search(self, query: str) -> Optional[WebSearchResult]:
        """Search web and return the top result only."""
        try:
            print(f"Searching web for: {query}")
            
            # Try DuckDuckGo HTML search for the top result
            result = self._search_ddg_html(query)
            if result:
                print(f"Found top result: {result.title}")
                return result
            
            print("No search results found")
            return None
            
        except Exception as e:
            print(f"Web search error: {e}")
            return None

    def _search_ddg_html(self, query: str) -> Optional[WebSearchResult]:
        """Search using DuckDuckGo and return the first valid result."""
        try:
            url = "https://duckduckgo.com/html/"
            params = {'q': query}
            
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                html = response.text
                
                # Pattern to find the first result with title, URL, and snippet
                pattern = r'<h2[^>]*class="[^"]*result[^"]*"[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>([^<]+)</a>.*?</h2>.*?<span[^>]*class="[^"]*result-snippet[^"]*"[^>]*>([^<]+)</span>'
                
                match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
                
                if match:
                    url_found, title, snippet = match.groups()
                    
                    # Clean the data
                    url_found = self._clean_url(url_found)
                    title = self._clean_text(title)
                    snippet = self._clean_text(snippet)
                    
                    # Validate and return the first good result
                    if (self._is_valid_url(url_found) and 
                        len(title) > 3 and 
                        len(snippet) > 5):
                        
                        return WebSearchResult(
                            title=title,
                            url=url_found,
                            snippet=snippet[:200] + "..." if len(snippet) > 200 else snippet
                        )
            
            return None
            
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return None

    def _clean_url(self, url: str) -> str:
        """Clean URL from DuckDuckGo redirects."""
        if not url:
            return ""
        
        # Handle DuckDuckGo redirects
        if '/l/?uddg=' in url:
            try:
                decoded = unquote(url)
                match = re.search(r'https?://[^&\s]+', decoded)
                if match:
                    return match.group(0)
            except:
                pass
        
        # Fix relative URLs
        if url.startswith('//'):
            url = 'https:' + url
        elif url.startswith('/'):
            url = 'https://duckduckgo.com' + url
        
        return url.strip()

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Decode HTML entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&quot;', '"').replace('&#39;', "'")
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and not an internal link."""
        if not url or not url.startswith('http'):
            return False
        
        # Skip unwanted domains
        unwanted = ['duckduckgo.com', 'google.com', 'bing.com']
        for domain in unwanted:
            if domain in url.lower():
                return False
        
        return len(url) > 10