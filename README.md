# Course Script + Audio + YouTube Metadata (Streamlit)

This app converts course URLs into a 3-step workflow:

1. Generate 2-3 minute spoken scripts from course URLs with Gemini.
2. Generate WAV voiceover for each script with Gemini TTS using a Google service account.
3. Generate YouTube title/description/tags using the script plus computed audio duration.

## Setup

### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Gemini
Set `GEMINI_API_KEY` using Streamlit secrets or environment variables.

`.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "YOUR_KEY"
GCP_TTS_SERVICE_ACCOUNT_JSON = """{"type":"service_account",...}"""
```

### Gemini TTS service account
The audio step now uses the same service-account flow as `demo tts/google_tts_service_account.py`.

You can provide the service account in the sidebar by:
- Uploading the JSON file
- Pasting the raw JSON
- Entering a local file path

The service account needs permission to call Google Cloud Text-to-Speech.

## Run
```bash
streamlit run app.py
```

## Notes
- Audio exports are WAV files.
- The YouTube metadata step needs the generated audio bytes to compute duration, so the easiest flow is still 1 -> 2 -> 3 in one session.
- If you move between sessions, upload the audio CSV plus the ZIP from Tab 2.
