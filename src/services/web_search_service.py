# src/services/web_search_service.py
"""Web search service with multiple fallback options for Streamlit Cloud."""

from typing import Optional
from dataclasses import dataclass
import sys
from pathlib import Path
import time
import random

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

@dataclass
class WebSearchResult:
    """Web search result."""
    title: str
    url: str
    snippet: str

class WebSearchService:
    """Robust web search service with multiple fallback options."""

    def __init__(self):
        """Initialize web search service with multiple options."""
        self.search_tools = []
        self._initialize_search_tools()
        
        # User agents for rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]

    def _initialize_search_tools(self):
        """Initialize available search tools in order of preference."""
        
        # Method 1: DuckDuckGo Search (most reliable)
        try:
            from duckduckgo_search import DDGS
            self.search_tools.append(('ddgs', DDGS()))
        except ImportError:
            pass
        
        # Method 2: LangChain DuckDuckGo
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            self.search_tools.append(('langchain_ddg', DuckDuckGoSearchRun()))
        except ImportError:
            pass
        
        # Method 3: Requests-based fallback
        try:
            import requests
            self.requests = requests
            self.search_tools.append(('requests', 'requests_fallback'))
        except ImportError:
            pass

    def search(self, query: str, max_results: int = 3) -> Optional[WebSearchResult]:
        """Search web using available tools with fallbacks."""
        # Try each search tool in order
        for tool_name, tool in self.search_tools:
            try:
                result = self._search_with_tool(tool_name, tool, query, max_results)
                if result:
                    return result
            except Exception:
                continue
        
        return None

    def _search_with_tool(self, tool_name: str, tool, query: str, max_results: int) -> Optional[WebSearchResult]:
        """Search with a specific tool."""
        
        if tool_name == 'ddgs':
            return self._search_with_ddgs(tool, query, max_results)
        elif tool_name == 'langchain_ddg':
            return self._search_with_langchain(tool, query)
        elif tool_name == 'requests':
            return self._search_with_requests(query)
        
        return None

    def _search_with_ddgs(self, ddgs, query: str, max_results: int) -> Optional[WebSearchResult]:
        """Search using duckduckgo-search library (most reliable)."""
        try:
            # Add small delay to avoid rate limiting
            time.sleep(random.uniform(0.5, 1.5))
            
            # Use the text search method
            results = list(ddgs.text(
                keywords=query,
                max_results=max_results,
                region='wt-wt',  # Global search
                safesearch='moderate',
                timelimit=None
            ))
            
            if results:
                # Get the first result
                result = results[0]
                return WebSearchResult(
                    title=self._clean_text(result.get('title', '')),
                    url=result.get('href', ''),
                    snippet=self._clean_text(result.get('body', ''))
                )
            
            return None
            
        except Exception:
            return None

    def _search_with_langchain(self, tool, query: str) -> Optional[WebSearchResult]:
        """Search using LangChain DuckDuckGo."""
        try:
            # Add delay
            time.sleep(random.uniform(0.5, 1.0))
            
            search_results = tool.run(query)
            
            if search_results and len(search_results) > 50:
                # Parse the results
                return self._parse_langchain_results(search_results, query)
            
            return None
            
        except Exception:
            return None

    def _search_with_requests(self, query: str) -> Optional[WebSearchResult]:
        """Fallback search using requests."""
        try:
            import requests
            from urllib.parse import quote
            
            # Add delay
            time.sleep(random.uniform(1.0, 2.0))
            
            # Use DuckDuckGo instant answer API
            headers = {
                'User-Agent': random.choice(self.user_agents)
            }
            
            # Try DuckDuckGo instant answer API first
            instant_url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1"
            
            response = requests.get(instant_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for instant answer
                if data.get('AbstractText') and data.get('AbstractURL'):
                    return WebSearchResult(
                        title=data.get('Heading', query),
                        url=data.get('AbstractURL', ''),
                        snippet=self._clean_text(data.get('AbstractText', ''))
                    )
                
                # Check for related topics
                if data.get('RelatedTopics'):
                    for topic in data.get('RelatedTopics', [])[:1]:
                        if isinstance(topic, dict) and topic.get('Text') and topic.get('FirstURL'):
                            return WebSearchResult(
                                title=topic.get('Text', query)[:100],
                                url=topic.get('FirstURL', ''),
                                snippet=self._clean_text(topic.get('Text', ''))
                            )
            
            return None
            
        except Exception:
            return None

    def _parse_langchain_results(self, search_results: str, query: str) -> Optional[WebSearchResult]:
        """Parse LangChain search results."""
        try:
            lines = [line.strip() for line in search_results.split('\n') if line.strip()]
            
            title = ""
            url = ""
            snippet = ""
            
            # Look for patterns in the results
            import re
            
            # Extract URLs
            urls = re.findall(r'https?://[^\s]+', search_results)
            if urls:
                url = urls[0]
            
            # Get first meaningful line as title
            for line in lines:
                if len(line) > 10 and not line.startswith('http') and not any(char in line for char in ['[', ']', '{']):
                    title = line
                    break
            
            # Get longer text as snippet
            for line in lines:
                if len(line) > 50 and line != title:
                    snippet = line
                    break
            
            if title and url:
                return WebSearchResult(
                    title=self._clean_text(title),
                    url=url,
                    snippet=self._clean_text(snippet) if snippet else self._clean_text(title)
                )
            
            return None
            
        except Exception:
            return None

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        import re
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Decode HTML entities
        html_entities = {
            '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"',
            '&#39;': "'", '&nbsp;': ' ', '&apos;': "'", '&copy;': '©'
        }
        for entity, char in html_entities.items():
            text = text.replace(entity, char)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\-.,!?()&:;/]', '', text)
        
        return text.strip()