# create a fresh one
python -m venv .venv

# activate it
.\.venv\Scripts\Activate.ps1       # Windows PowerShell

# Install dependencies
pip install --upgrade pip
# Cancel the current slow install (Ctrl+C)
# Then run:
pip install "chromadb==0.4.24" "pyarrow<15" "duckdb<0.9.0"
pip install -r requirements.txt

# Run your scripts
python ingest_pdfs.py
python build_index.py
python chatbot.py cli