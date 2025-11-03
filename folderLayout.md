📁 Folder Layout
medicare_chatbot/
├── ingest_pdfs.py          # extracts & chunks plan PDFs
├── build_index.py          # embeds text into a vector DB
├── chatbot.py              # interactive QA app (CLI or web)
├── requirements.txt
├── data/
│   ├── pdfs/               # put your PDFs here
│   ├── texts/              # auto-generated plain-text dumps
│   └── index/              # Chroma/FAISS vector store
└── .env                    # optional, holds your OpenAI key

⚙️ Workflow Summary
# 1. Place your PDFs
mkdir -p data/pdfs
# 2. Extract & chunk
python ingest_pdfs.py
# 3. Build embeddings
python build_index.py
# 4. Chat!
python chatbot.py cli
# or
streamlit run chatbot.py

🧠 Next upgrades
Goal	How
Faster / cheaper embedding	Replace OpenAIEmbeddings with SentenceTransformerEmbeddings('all-MiniLM-L6-v2')
Better citations	Modify RetrievalQA chain to include source_documents in output
Fine-tuned summarizer	Add post-processing step using a smaller LLM for summaries
Deploy online	Containerize with Docker + FastAPI endpoint for cloud or local LAN use
Offline use	Swap ChatOpenAI with a local model via llama-cpp-python