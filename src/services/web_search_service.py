# src/services/web_search_service.py
"""Improved web search service with multiple fallback options for Streamlit Cloud."""

from typing import Optional, List
from dataclasses import dataclass
import sys
from pathlib import Path
import json
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
            print("✅ DuckDuckGo Search (ddgs) initialized")
        except ImportError:
            print("⚠️ duckduckgo-search not available")
        
        # Method 2: LangChain DuckDuckGo
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            self.search_tools.append(('langchain_ddg', DuckDuckGoSearchRun()))
            print("✅ LangChain DuckDuckGo initialized")
        except ImportError:
            print("⚠️ LangChain DuckDuckGo not available")
        
        # Method 3: Requests-based fallback
        try:
            import requests
            self.requests = requests
            self.search_tools.append(('requests', 'requests_fallback'))
            print("✅ Requests fallback initialized")
        except ImportError:
            print("⚠️ Requests not available")
        
        if not self.search_tools:
            print("❌ No search tools available!")
        else:
            print(f"✅ Initialized {len(self.search_tools)} search methods")

    def search(self, query: str, max_results: int = 3) -> Optional[WebSearchResult]:
        """Search web using available tools with fallbacks."""
        print(f"🔍 Searching for: '{query}'")
        
        # Try each search tool in order
        for tool_name, tool in self.search_tools:
            print(f"🔄 Trying {tool_name}...")
            
            try:
                result = self._search_with_tool(tool_name, tool, query, max_results)
                if result:
                    print(f"✅ Success with {tool_name}: {result.title[:50]}...")
                    return result
                else:
                    print(f"⚠️ {tool_name} returned no results")
            except Exception as e:
                print(f"❌ {tool_name} failed: {e}")
                continue
        
        print("❌ All search methods failed")
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
            
        except Exception as e:
            print(f"DDGS search error: {e}")
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
            
        except Exception as e:
            print(f"LangChain search error: {e}")
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
            
        except Exception as e:
            print(f"Requests search error: {e}")
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
            
        except Exception as e:
            print(f"Error parsing LangChain results: {e}")
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

    def test_search_tools(self) -> dict:
        """Test all available search tools."""
        test_query = "artificial intelligence"
        results = {}
        
        for tool_name, tool in self.search_tools:
            try:
                print(f"Testing {tool_name}...")
                result = self._search_with_tool(tool_name, tool, test_query, 1)
                results[tool_name] = {
                    'status': 'success' if result else 'no_results',
                    'result': result.title[:50] + "..." if result else None
                }
            except Exception as e:
                results[tool_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results

def main():
    """Test the web search service."""
    print("Testing Web Search Service")
    print("=" * 40)
    
    service = WebSearchService()
    
    # Test search tools
    print("\n1. Testing search tools...")
    test_results = service.test_search_tools()
    for tool, result in test_results.items():
        status = result['status']
        if status == 'success':
            print(f"✅ {tool}: {result['result']}")
        elif status == 'no_results':
            print(f"⚠️ {tool}: No results")
        else:
            print(f"❌ {tool}: {result.get('error', 'Unknown error')}")
    
    # Test actual search
    print("\n2. Testing actual search...")
    test_queries = [
        "liderlik becerileri",
        "leadership skills",
        "artificial intelligence trends"
    ]
    
    for query in test_queries:
        print(f"\nSearching: '{query}'")
        result = service.search(query)
        if result:
            print(f"✅ Found: {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Snippet: {result.snippet[:100]}...")
        else:
            print("❌ No results found")

if __name__ == "__main__":
    main()