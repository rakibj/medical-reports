from typing import List
from openai import OpenAI
import re


class TextEmbedder:
    def __init__(self):
        self.client = OpenAI()

    # ---- tiny splitter (roughly 700–900 tokens per chunk if English-like)
    def simple_chunk(self, text: str, max_chars: int = 3500, overlap: int = 200) -> List[str]:
        text = re.sub(r'\s+', ' ', text).strip()
        chunks = []
        i = 0
        while i < len(text):
            end = min(i + max_chars, len(text))
            # try to break at sentence boundary near the end
            cut = text.rfind(". ", i, end)
            if cut == -1 or cut < i + int(max_chars * 0.6):
                cut = end
            else:
                cut += 1  # include the period
            chunks.append(text[i:cut].strip())
            i = max(cut - overlap, cut)  # ensure progress if overlap > chunk
        return [c for c in chunks if c]


    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        EMBED_MODEL = "text-embedding-3-small"  # 1536 dims
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [d.embedding for d in resp.data]
    
    def embed_query(self, q: str) -> list[float]:
        return self.client.embeddings.create(
            model="text-embedding-3-small", input=[q]
        ).data[0].embedding