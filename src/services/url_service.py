from urllib.parse import urlparse
from typing import List


class URLService:
    @staticmethod
    def extract_domain(url: str) -> str:
        """Extract domain from URL."""
        return urlparse(url).netloc
    
    @staticmethod
    def update_url_list(urls: List[str], used_domains: List[str]) -> tuple[List[str], List[str]]:
        """Remove first URL from list and track its domain."""
        if not urls:
            return urls, used_domains
            
        url = urls[0]
        domain = URLService.extract_domain(url)
        
        # Track used domains
        updated_used_domains = used_domains.copy()
        if domain not in updated_used_domains:
            updated_used_domains.append(domain)
        
        # Remove first URL
        remaining_urls = urls[1:]
        
        return remaining_urls, updated_used_domains