# tools/build_local_index.py
import os, json, pathlib, re
import numpy as np
import requests

# Avoid TF/Flax imports from transformers (we only need PyTorch)
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence TF logs just in case

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

URLS = [
    "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters",
    "https://www.cdc.gov/coronavirus/2019-ncov/vaccines/facts.html",
    "https://www.snopes.com/collections/coronavirus-collection/",
    "https://www.politifact.com/coronavirus/",
    "https://fullfact.org/health/coronavirus/",
    "https://www.reuters.com/fact-check/health/",
]

SAVE_DIR = pathlib.Path("App/fact_index")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def extract_text(url: str) -> str:
    """Try trafilatura for clean text, fallback to simple HTML strip."""
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url, timeout=20)
        if downloaded:
            txt = trafilatura.extract(downloaded, include_comments=False) or ""
            if len(txt) > 500:
                return txt
    except Exception:
        pass
    try:
        html = requests.get(url, timeout=20).text
        txt = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", html, flags=re.I)
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt
    except Exception:
        return ""

def main():
    docs = []
    for u in URLS:
        print("Fetching:", u)
        txt = extract_text(u)
        if len(txt) > 1000:
            title = u.split("//", 1)[-1][:140]
            docs.append({"url": u, "title": title, "text": txt})
    if not docs:
        print("No docs extracted. Add more URLs or check your internet.")
        return

    print(f"Embedding {len(docs)} pages …")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode([d["text"][:4000] for d in docs],
                       normalize_embeddings=True, show_progress_bar=True)

    np.save(SAVE_DIR / "embeddings.npy", emb)
    with open(SAVE_DIR / "docs.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    nn = NearestNeighbors(n_neighbors=10, metric="cosine")
    nn.fit(emb)

    import joblib
    joblib.dump(nn, SAVE_DIR / "nn.joblib")
    print("✅ Saved index to", SAVE_DIR.resolve())

if __name__ == "__main__":
    main()
