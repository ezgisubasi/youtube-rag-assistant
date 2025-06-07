# src/services/web_search_service.py
"""Web search service using DuckDuckGo search."""

from typing import Optional
from dataclasses import dataclass
import sys
from pathlib import Path
import requests
from urllib.parse import quote

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
        """Search web and return the best result."""
        try:
            print(f"Searching web for: {query}")
            
            # Try to get real search results using DuckDuckGo lite
            result = self._search_ddg_lite(query)
            
            if result:
                print(f"Found result: {result.title}")
                return result
            
            # If no results, return a Google search link as fallback
            print("No direct results found, returning Google search link")
            return WebSearchResult(
                title=f"Search results for: {query}",
                url=f"https://www.google.com/search?q={quote(query)}",
                snippet="Click to see search results on Google"
            )
            
        except Exception as e:
            print(f"Web search error: {e}")
            return WebSearchResult(
                title=f"Search: {query}",
                url=f"https://www.google.com/search?q={quote(query)}",
                snippet="Error during search. Click to search manually."
            )

    def _search_ddg_lite(self, query: str) -> Optional[WebSearchResult]:
        """Search using DuckDuckGo lite HTML interface."""
        try:
            # Use DuckDuckGo lite interface
            url = "https://lite.duckduckgo.com/lite/"
            params = {'q': query}
            
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                html = response.text
                
                # Simple regex to find the first real result
                # DuckDuckGo lite has a simple structure
                import re
                
                # Pattern: Find links that are actual results (not DuckDuckGo internal)
                # Look for: <a class="result-link" href="URL">TITLE</a>
                pattern = r'<a[^>]*class="result-link"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
                matches = re.findall(pattern, html)
                
                if not matches:
                    # Try alternative pattern for lite interface
                    pattern = r'<td><a[^>]*href="(https?://[^"]+)"[^>]*>([^<]+)</a>'
                    matches = re.findall(pattern, html)
                
                # Filter and return first good result
                for found_url, title in matches:
                    # Clean URL from DuckDuckGo redirects
                    clean_url = self._clean_url(found_url)
                    
                    # Skip DuckDuckGo internal links and invalid URLs
                    if (self._is_valid_url(clean_url) and 
                        len(title.strip()) > 10):  # Ensure it's a real title
                        
                        # Try to extract snippet
                        snippet = self._extract_snippet(html, found_url, title)
                        
                        return WebSearchResult(
                            title=title.strip(),
                            url=clean_url,
                            snippet=snippet
                        )
            
            return None
            
        except Exception as e:
            print(f"DuckDuckGo lite search error: {e}")
            return None
    
    def _clean_url(self, url: str) -> str:
        """Clean URL from DuckDuckGo redirects."""
        # Remove DuckDuckGo redirect wrappers
        if url.startswith('/l/?kh=-1&uddg='):
            try:
                import urllib.parse
                decoded = urllib.parse.unquote(url)
                real_url_match = re.search(r'https?://[^&]+', decoded)
                if real_url_match:
                    return real_url_match.group(0)
            except:
                pass
        
        # Fix relative URLs
        if url.startswith('//'):
            url = 'https:' + url
        elif url.startswith('/'):
            url = 'https://duckduckgo.com' + url
        
        return url.strip()
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and not an internal link."""
        if not url or not url.startswith('http'):
            return False
        
        # Skip unwanted domains
        unwanted = ['duckduckgo.com', 'google.com/search', 'bing.com', 'yahoo.com']
        for domain in unwanted:
            if domain in url.lower():
                return False
        
        return len(url) > 10
    
    def _extract_snippet(self, html: str, url: str, title: str) -> str:
        """Try to extract a snippet for the result."""
        try:
            # In DuckDuckGo lite, snippets are usually in the next table cell
            import re
            
            # Escape special regex characters in URL
            escaped_url = re.escape(url)
            
            # Look for text after the link
            pattern = rf'{escaped_url}[^<]*</a>[^<]*</td>[^<]*<td[^>]*>([^<]+)'
            match = re.search(pattern, html)
            
            if match:
                snippet = match.group(1).strip()
                # Clean up the snippet
                snippet = re.sub(r'\s+', ' ', snippet)  # Remove extra whitespace
                if len(snippet) > 20:  # Ensure it's meaningful
                    return snippet[:200] + "..." if len(snippet) > 200 else snippet
            
            # Fallback snippet - just the content without "Search result for"
            return title
            
        except:
            return title
        
        