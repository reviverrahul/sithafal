import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    """
    Scrape the text content from a given URL.
    
    Args:
        url (str): The URL to scrape.
    
    Returns:
        str: The scraped text content.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()  # Extract text from the HTML
    except requests.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return ""
