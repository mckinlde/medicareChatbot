import os, fitz, json
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter

PDF_DIR  = "data/pdfs"
TEXT_DIR = "data/texts"
os.makedirs(TEXT_DIR, exist_ok=True)

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": i+1, "text": text})
    return pages

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

for fname in tqdm(os.listdir(PDF_DIR)):
    if not fname.lower().endswith(".pdf"): continue
    pdf_path = os.path.join(PDF_DIR, fname)
    plan_id  = fname.split("_")[1] if "_" in fname else fname
    pages = extract_text(pdf_path)
    chunks_out = []

    for p in pages:
        chunks = splitter.split_text(p["text"])
        for i, c in enumerate(chunks):
            chunks_out.append({
                "plan_id": plan_id,
                "page": p["page"],
                "chunk_id": i,
                "text": c,
                "source": fname
            })

    out_path = os.path.join(TEXT_DIR, fname + ".jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks_out:
            f.write(json.dumps(c) + "\n")

print("✅ Text extracted to", TEXT_DIR)
