# setup and start virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install Ollama


# install Ollama dependencies
ollama pull nomic-embed-text

# install dependencies
pip install -r requirements.txt