import os, json, glob
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

TEXT_DIR = "data/texts"
INDEX_DIR = "data/index"
os.makedirs(INDEX_DIR, exist_ok=True)

embeddings = OpenAIEmbeddings()  # or use SentenceTransformerEmbeddings()

texts, metas = [], []
for fp in tqdm(glob.glob(os.path.join(TEXT_DIR, "*.jsonl"))):
    with open(fp, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            texts.append(d["text"])
            metas.append({
                "plan_id": d["plan_id"],
                "page": d["page"],
                "source": d["source"]
            })

print(f"Embedding {len(texts)} chunks...")
vectordb = Chroma.from_texts(texts, embeddings, metadatas=metas,
                             persist_directory=INDEX_DIR)
vectordb.persist()
print("✅ Index built at", INDEX_DIR)
