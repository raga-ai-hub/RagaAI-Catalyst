"""
Script to fetch, download, and summarize the most upvoted paper from Hugging Face daily papers.
This script uses SmoLAgents to create a pipeline that:
1. Fetches the top paper from Hugging Face
2. Gets its arXiv ID
3. Downloads the paper
4. Reads and summarizes its content
"""


import json
import arxiv
import requests
from bs4 import BeautifulSoup
from huggingface_hub import HfApi
from pypdf import PdfReader
from smolagents import CodeAgent, HfApiModel, tool

@tool
def get_hugging_face_top_daily_paper() -> str:
    """
    Fetch the most upvoted paper on Hugging Face daily papers.

    Returns:
        str: The title of the most upvoted paper, or None if an error occurs
    """
    try:
        url = "https://huggingface.co/papers"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        containers = soup.find_all('div', class_='SVELTE_HYDRATER contents')
        
        for container in containers:
            data_props = container.get('data-props', '')
            if not data_props:
                continue
                
            try:
                json_data = json.loads(data_props.replace('&quot;', '"'))
                if 'dailyPapers' in json_data and json_data['dailyPapers']:
                    return json_data['dailyPapers'][0]['title']
            except json.JSONDecodeError:
                continue
                
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching paper from Hugging Face: {e}")
        return None

@tool
def get_paper_id_by_title(title: str) -> str:
    """
    Get the arXiv paper ID using its title.

    Args:
        title (str): The paper title to search for

    Returns:
        str: The arXiv paper ID, or None if not found
    """
    if not title:
        return None
        
    try:
        api = HfApi()
        papers = api.list_papers(query=title)
        return next(iter(papers)).id if papers else None
    except Exception as e:
        print(f"Error getting paper ID: {e}")
        return None

@tool
def download_paper_by_id(paper_id: str) -> bool:
    """
    Download a paper from arXiv using its ID.

    Args:
        paper_id (str): The arXiv paper ID

    Returns:
        bool: True if download successful, False otherwise
    """
    if not paper_id:
        return False
        
    try:
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[paper_id])))
        paper.download_pdf(filename="paper.pdf")
        return True
    except Exception as e:
        print(f"Error downloading paper: {e}")
        return False

@tool
def read_pdf_file(file_path: str = "paper.pdf") -> str:
    """
    Read the first three pages of a PDF file.

    Args:
        file_path (str): Path to the PDF file, defaults to 'paper.pdf'

    Returns:
        str: Extracted text content from the first three pages
    """
    try:
        reader = PdfReader(file_path)
        content = ""
        
        # Get number of pages for logging
        total_pages = len(reader.pages)
        print(f"Total pages in PDF: {total_pages}")
        
        # Read first three pages or all pages if less than three
        pages_to_read = min(3, total_pages)
        for page in reader.pages[:pages_to_read]:
            content += page.extract_text()
            
        return content
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def main():
    """Initialize and run the paper summarization agent."""
    model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
    model = HfApiModel(model_id=model_id, token='HF_API_TOKEN')
    
    agent = CodeAgent(
        tools=[
            get_hugging_face_top_daily_paper,
            get_paper_id_by_title,
            download_paper_by_id,
            read_pdf_file
        ],
        model=model,
        add_base_tools=True
    )

    agent.run(
        "Summarize today's top paper on Hugging Face daily papers by reading it."
    )

if __name__ == "__main__":
    main()