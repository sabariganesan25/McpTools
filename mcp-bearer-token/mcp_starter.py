import asyncio
from typing import Annotated
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl

import markdownify
import httpx
import readabilipy

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        """
        Perform a scoped DuckDuckGo search and return a list of job posting URLs.
        (Using DuckDuckGo because Google blocks most programmatic scraping.)
        """
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []

        async with httpx.AsyncClient() as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Failed to perform search.</error>"]

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__a", href=True):
            href = a["href"]
            if "http" in href:
                links.append(href)
            if len(links) >= num_results:
                break

        return links or ["<error>No results found.</error>"]

# --- MCP Server Setup ---
mcp = FastMCP(
    "Job Finder MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER



# --- Tool: github_project_finder ---
GitHubProjectFinderDescription = RichToolDescription(
    description="Find GitHub repositories related to a given topic or keyword.",
    use_when="Use this when you need open-source projects from GitHub on a given subject.",
    side_effects="Returns a list of GitHub repository links relevant to the search topic."
)

@mcp.tool(description=GitHubProjectFinderDescription.model_dump_json())
async def github_project_finder(
    topic: Annotated[str, Field(description="Topic or keyword to search GitHub for relevant repositories")],
    num_results: Annotated[int, Field(description="Number of GitHub repository links to return (default 5)")] = 5
) -> str:
    """
    Searches GitHub for repositories related to the provided topic and returns their links.
    """
    query = f"site:github.com {topic} repository"
    links = await Fetch.google_search_links(query, num_results=num_results)

    # Filter to GitHub links only
    github_links = [link for link in links if "github.com" in link.lower()]

    if not github_links:
        return f"❌ No GitHub repositories found for topic: **{topic}**"

    return (
        f"📂 **GitHub Projects for**: _{topic}_\n\n" +
        "\n".join(f"- {link}" for link in github_links)
    )
# --- Load YouTube API Key ---
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
assert YOUTUBE_API_KEY is not None, "Please set YOUTUBE_API_KEY in your .env file"

# --- Tool: youtube_video_finder (YouTube API Version) ---
YouTubeVideoFinderDescription = RichToolDescription(
    description="Find YouTube videos related to a given topic or keyword using YouTube Data API v3.",
    use_when="Use this when you need relevant YouTube video links for a given topic.",
    side_effects="Returns a list of YouTube video titles and links."
)

@mcp.tool(description=YouTubeVideoFinderDescription.model_dump_json())
async def youtube_video_finder(
    topic: Annotated[str, Field(description="Topic or keyword to search YouTube for relevant videos")],
    num_results: Annotated[int, Field(description="Number of YouTube videos to return (default 5)")] = 5
) -> str:
    """
    Searches YouTube for videos related to the provided topic using YouTube Data API v3.
    """
    api_url = (
        "https://www.googleapis.com/youtube/v3/search"
        f"?part=snippet&q={topic}&type=video&maxResults={num_results}&key={YOUTUBE_API_KEY}"
    )

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(api_url, timeout=30)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch from YouTube API: {e!r}"))

    data = resp.json()
    if "items" not in data or not data["items"]:
        return f"❌ No YouTube videos found for topic: **{topic}**"

    videos_list = []
    for item in data["items"]:
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        channel = item["snippet"]["channelTitle"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        videos_list.append(f"▶️ **{title}** (by _{channel}_) → {url}")

    return f"📺 **YouTube Videos for**: _{topic}_\n\n" + "\n".join(videos_list)



# --- Load Google API credentials ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
assert GOOGLE_API_KEY is not None, "Please set GOOGLE_API_KEY in your .env file"
assert GOOGLE_CSE_ID is not None, "Please set GOOGLE_CSE_ID in your .env file"

# --- Tool: pdf_document_finder_google ---
PdfDocumentFinderGoogleDescription = RichToolDescription(
    description="Find PDF and document files related to a given topic using Google Custom Search API.",
    use_when="Use this when you need PDFs or docs (filetype:pdf/doc/docx) on a subject.",
    side_effects="Returns a list of Google search results filtered for document file types."
)

@mcp.tool(description=PdfDocumentFinderGoogleDescription.model_dump_json())
async def pdf_document_finder_google(
    topic: Annotated[str, Field(description="Topic or keyword to search for PDF/docs")],
    num_results: Annotated[int, Field(description="Number of document links to return (default 5)")] = 5
) -> str:
    """
    Search Google Custom Search API for PDFs/docs on `topic` and return top links.
    """
    search_query = f"{topic} (filetype:pdf OR filetype:doc OR filetype:docx OR filetype:ppt OR filetype:pptx)"
    api_url = (
        f"https://www.googleapis.com/customsearch/v1"
        f"?key={GOOGLE_API_KEY}"
        f"&cx={GOOGLE_CSE_ID}"
        f"&q={search_query}"
        f"&num={num_results}"
    )

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(api_url, timeout=30)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Google API fetch failed: {e!r}"))

    data = resp.json()
    if "items" not in data or not data["items"]:
        return f"❌ No document results found for topic: **{topic}**"

    doc_links = []
    for item in data["items"]:
        link = item.get("link", "")
        if link.lower().endswith((".pdf", ".doc", ".docx", ".ppt", ".pptx")):
            doc_links.append(f"- {link}")

    if not doc_links:
        return f"⚠️ No direct document links found for **{topic}**, but search returned results."

    return f"📄 **Documents for**: _{topic}_\n\n" + "\n".join(doc_links)



# --- Run MCP Server ---
async def main():
    print("Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
