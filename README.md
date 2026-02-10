# Course Script + Audio + YouTube Metadata (Streamlit)

This app converts your notebook into a multi-step Streamlit UI:

1. Generate 2–3 minute spoken scripts from course URLs (Gemini).
2. Generate MP3 voiceover for each script (Google Cloud Text-to-Speech).
3. Generate YouTube title/description/tags using script + audio duration (Gemini).

## Setup

### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Gemini
Set `GEMINI_API_KEY` using Streamlit secrets or environment variables.

`.streamlit/secrets.toml` (create this file; do not commit):
```toml
GEMINI_API_KEY = "YOUR_KEY"
GCP_QUOTA_PROJECT = "optional-quota-project"
```

### Google Cloud TTS (ADC)
This app uses Application Default Credentials (ADC). Configure in your shell (local dev):

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_QUOTA_PROJECT
```

Or for deployment, use a service account JSON and set `GOOGLE_APPLICATION_CREDENTIALS`.

## Run
```bash
streamlit run app.py
```

## Notes
- Do **not** hardcode API keys in code.
- The YouTube metadata step needs the MP3 bytes to compute duration; easiest is to run steps 1→2→3 in one session, or upload the CSV + ZIP of MP3s.
