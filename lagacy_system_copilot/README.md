# PaySim Fraud Copilot

A local Streamlit assistant that analyzes the supplied `PaySim_Dataset.zip` financial-transaction dataset. It reads the CSV through Polars lazy queries—no PostgreSQL server, Docker database, or model-authored SQL is used. OpenAI API models provide the tool-calling assistant.

## Quick start

1. Keep `PaySim_Dataset.zip` in the project root, or set `PAYSIM_DATASET_PATH` in `.env` to a `.zip` or `.csv` path.
2. Copy `.env.example` to `.env`.
3. Create and activate a Python 3.11+ virtual environment, then run `pip install -r requirements.txt` and `pip install -e .`.
4. Add `OPENAI_API_KEY` to `.env`. Optionally change `OPENAI_MODEL`; the default is `gpt-4.1-mini`.
5. Start the UI: `streamlit run app.py`.

On the first query, the app extracts the archive once to ignored `data/PaySim_Dataset/`; later queries read the CSV lazily without loading all records into memory.

The included `workflow_dispatch` GitHub Action validates code and builds the container image. It deliberately does not SSH or deploy: this is a local dataset demo.

## Data and safety

PaySim uses `step` as an hourly simulation index, not a real-world date. The agent can preview raw transaction rows, calculate transaction metrics, assess fraud risk by type, find balance anomalies, and retrieve activity for an anonymized account. It cannot modify the source archive or execute arbitrary database queries.

The API key is read only by the Python process from `.env`; never put it in Streamlit client-side code or commit it to source control.
