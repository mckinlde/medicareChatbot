# create a fresh one
python -m venv .venv

# activate it
.\.venv\Scripts\activate       # Windows PowerShell

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run your scripts
python ingest_pdfs.py
python build_index.py
python chatbot.py cli