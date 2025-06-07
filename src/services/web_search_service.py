# src/services/web_search_service.py
"""Web search service using LangChain DuckDuckGo search."""

from typing import Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

@dataclass
class WebSearchResult:
    """Web search result."""
    title: str
    url: str
    snippet: str

class WebSearchService:
    """Web search service using LangChain DuckDuckGo."""

    def __init__(self):
        """Initialize web search service."""
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            self.search_tool = DuckDuckGoSearchRun()
            print("Web Search Service initialized with LangChain DuckDuckGo")
        except ImportError:
            print("Warning: LangChain DuckDuckGo not available, falling back to manual search")
            self.search_tool = None
            import requests
            self.requests = requests
            self.headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }

    def search(self, query: str) -> Optional[WebSearchResult]:
        """Search web and return the top result."""
        try:
            print(f"Searching web for: {query}")
            
            if self.search_tool:
                # Use LangChain DuckDuckGo
                result = self._search_with_langchain(query)
                if result:
                    print(f"Found result via LangChain: {result.title}")
                    return result
            
            # Fallback to manual search
            result = self._manual_search_fallback(query)
            if result:
                print(f"Found result via fallback: {result.title}")
                return result
            
            print("No search results found")
            return None
            
        except Exception as e:
            print(f"Web search error: {e}")
            return None

    def _search_with_langchain(self, query: str) -> Optional[WebSearchResult]:
        """Search using LangChain DuckDuckGo tool."""
        try:
            # Get search results from LangChain
            search_results = self.search_tool.run(query)
            
            if search_results and len(search_results) > 50:  # Ensure we got meaningful results
                # Parse the search results text
                lines = search_results.split('\n')
                
                title = ""
                url = ""
                snippet = ""
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Look for title (usually first meaningful line)
                    if not title and len(line) > 10 and not line.startswith('http'):
                        title = line
                    
                    # Look for URL
                    elif line.startswith('http') and not url:
                        url = line
                    
                    # Look for snippet (content after title and URL)
                    elif title and url and len(line) > 20:
                        snippet = line
                        break
                
                # If we found good components, return result
                if title and url and snippet:
                    return WebSearchResult(
                        title=self._clean_text(title),
                        url=url.strip(),
                        snippet=self._clean_text(snippet)[:300] + "..." if len(snippet) > 300 else self._clean_text(snippet)
                    )
                
                # Alternative parsing: treat entire result as snippet and extract URL
                elif search_results:
                    # Extract first URL from the text
                    import re
                    urls = re.findall(r'https?://[^\s]+', search_results)
                    if urls:
                        url = urls[0]
                        # Use the first meaningful line as title
                        first_line = search_results.split('\n')[0].strip()
                        title = first_line if len(first_line) > 10 else query
                        # Use first 200 chars as snippet
                        snippet = search_results[:200] + "..."
                        
                        return WebSearchResult(
                            title=self._clean_text(title),
                            url=url,
                            snippet=self._clean_text(snippet)
                        )
            
            return None
            
        except Exception as e:
            print(f"LangChain DuckDuckGo search error: {e}")
            return None

    def _manual_search_fallback(self, query: str) -> Optional[WebSearchResult]:
        """Fallback manual search if LangChain fails."""
        try:
            if not hasattr(self, 'requests'):
                return None
                
            # Simple DuckDuckGo lite search
            url = "https://lite.duckduckgo.com/lite/"
            params = {'q': query}
            
            response = self.requests.get(url, params=params, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                html = response.text
                import re
                
                # Simple pattern for DuckDuckGo lite
                pattern = r'<td><a[^>]*href="(https?://[^"]+)"[^>]*>([^<]+)</a></td><td>([^<]+)</td>'
                match = re.search(pattern, html, re.DOTALL)
                
                if match:
                    url_found, title, snippet = match.groups()
                    
                    # Clean and validate
                    url_found = self._clean_url(url_found)
                    title = self._clean_text(title)
                    snippet = self._clean_text(snippet)
                    
                    if self._is_valid_url(url_found) and len(title) > 3:
                        return WebSearchResult(
                            title=title,
                            url=url_found,
                            snippet=snippet[:300] + "..." if len(snippet) > 300 else snippet
                        )
            
            return None
            
        except Exception as e:
            print(f"Manual search fallback error: {e}")
            return None

    def _clean_url(self, url: str) -> str:
        """Clean URL from redirects."""
        if not url:
            return ""
        
        # Handle DuckDuckGo redirects
        if '/l/?uddg=' in url:
            try:
                from urllib.parse import unquote
                import re
                decoded = unquote(url)
                match = re.search(r'https?://[^&\s"\']+', decoded)
                if match:
                    return match.group(0)
            except:
                pass
        
        return url.strip()

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        import re
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Decode HTML entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        if not url or not url.startswith('http'):
            return False
        
        # Skip unwanted domains
        unwanted = ['duckduckgo.com', 'google.com', 'bing.com']
        for domain in unwanted:
            if domain in url.lower():
                return False
        
        return len(url) > 15 and '.' in url
    
    