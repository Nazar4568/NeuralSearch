import sys
import os
import httpx
import asyncio


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data.loader import DocumentLoader


async def index_file(file_path: str, api_url: str = "http://localhost:8000/index"):
    loader = DocumentLoader()

    absolute_path = os.path.join(project_root, file_path)

    try:
        chunks = loader.load_and_chunk(file_path=absolute_path, chunk_size=128, overlap=16)
    except FileNotFoundError:
        print(f"File not found: {absolute_path}")
        return

    if not chunks:
        print("No valid text found to index.")
        return

    print(f"Prepared {len(chunks)} chunks. Sending to API...")

    payload = {
        "documents": chunks,
        "collection_name": "documents"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(api_url, json=payload)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")
        except httpx.RequestError as e:
            print(f"Network error: {e}")


if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.txt"
    asyncio.run(index_file(file_path))