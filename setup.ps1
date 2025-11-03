# remove the broken env
Remove-Item -Recurse -Force .venv

# make sure python 3.11 or 3.12 is installed
py -3.12 --version       # or py -3.12 --version

# create new env
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# install pinned deps
pip install --upgrade pip
pip install "chromadb==0.4.24" "pyarrow<15" "duckdb<0.9.0"
pip install -r requirements.txt


# Run your scripts
python ingest_pdfs.py
python build_index.py
python chatbot.py cli