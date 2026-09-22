.PHONY: setup update notebook kaggle-auth download-titanic download-aita

# Create the virtualenv and install exactly what uv.lock pins.
setup:
	uv sync --frozen

# Re-resolve dependencies to their latest allowed versions,
# refresh uv.lock, and apply it to the virtualenv.
update:
	uv lock --upgrade
	uv sync

# Open the marimo notebook in a browser. --watch picks up edits made in an editor.
notebook:
	uv run marimo edit notebook.py --watch

# Log in to Kaggle (OAuth, opens a browser). Credentials are cached locally.
kaggle-auth:
	uv run kaggle auth login

# Download the AITA labeled posts dataset from Hugging Face into data/aita/
download-aita:
	uv run src/download.py -d nicoco404/AITA_labeled_posts -n aita
