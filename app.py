\
import io
import zipfile
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import google.auth

from src.prompts import SYSTEM_INSTRUCTION_LONG, SYSTEM_METADATA
from src.utils import dedupe_preserve_order
from src.gemini_client import get_client as get_gemini_client
from src.tts_client import get_tts_client
from src.pipeline import (
    run_script_generation,
    run_tts_generation,
    run_youtube_metadata,
)

from google.cloud import texttospeech

st.set_page_config(page_title="Course Script + Audio + YouTube Metadata", layout="wide")
st.title("Course Script Generator (Gemini) + Audio (Google TTS) + YouTube Metadata")

# ----------------------------
# Session state
# ----------------------------
if "df_scripts" not in st.session_state:
    st.session_state["df_scripts"] = None
if "df_audio" not in st.session_state:
    st.session_state["df_audio"] = None
if "audio_blobs" not in st.session_state:
    st.session_state["audio_blobs"] = []  # List[(filename, bytes)]
if "df_yt" not in st.session_state:
    st.session_state["df_yt"] = None
if "failures" not in st.session_state:
    st.session_state["failures"] = None

# ----------------------------
# Sidebar: keys + model config
# ----------------------------
with st.sidebar:
    st.header("Credentials")

    gemini_api_key = st.text_input(
        "Gemini API key (GEMINI_API_KEY)",
        type="password",
        value=st.secrets.get("GEMINI_API_KEY", ""),
        help="Use Streamlit secrets in deployment; don’t hardcode keys.",
    )

    st.header("Gemini models")
    gemini_script_model = st.text_input("Script model", value="gemini-2.5-pro")
    gemini_yt_model = st.text_input("YouTube metadata model", value="gemini-2.0-flash")

    st.header("Rate limits")
    min_seconds_scripts = st.slider("Seconds between script calls", 0.0, 10.0, 3.0, 0.5)
    min_seconds_yt = st.slider("Seconds between metadata calls", 0.0, 10.0, 1.5, 0.5)

    st.header("Google Cloud TTS (ADC)")
    st.caption("TTS uses Application Default Credentials (ADC) or a service account JSON provided below.")
    language_code = st.text_input("Language code", value="en-GB")
    with st.expander("TTS credentials (optional)", expanded=False):
        st.caption(
            "Provide either a service account JSON or an access token to avoid gcloud login.\n\n"
            "How to get a service account JSON:\n"
            "1) Google Cloud Console → IAM & Admin → Service Accounts\n"
            "2) Create Service Account and grant Text-to-Speech User role\n"
            "3) Open the service account → Keys → Add Key → Create new key → JSON\n"
            "4) Paste the JSON contents here"
        )
        sa_json = st.text_area(
            "Service account JSON",
            value=st.secrets.get("GCP_TTS_SERVICE_ACCOUNT_JSON", ""),
            height=140,
            placeholder='{"type":"service_account",...}',
        )
        access_token = st.text_input(
            "Access token",
            value=st.secrets.get("GCP_TTS_ACCESS_TOKEN", ""),
            type="password",
            help="Short-lived OAuth access token. Will expire.",
        )

    # Voice list (cached)
    @st.cache_data(show_spinner=False, ttl=3600)
    def list_voices(language_code: str, sa_json: str, access_token: str) -> List[str]:
        client = get_tts_client(
            service_account_json=sa_json or None,
            access_token=access_token or None,
        )
        voices = client.list_voices(language_code=language_code).voices
        return sorted({v.name for v in voices})

    enable_tts = st.toggle("Generate MP3", value=False)

    voice_name = ""
    model_name = ""
    style_prompt = ""

    if enable_tts:
        try:
            voice_options = list_voices(language_code, sa_json, access_token)
            voice_name = st.selectbox("Voice name", options=voice_options)
        except Exception as e:
            st.warning("Could not list voices. Check ADC / permissions.")
            st.exception(e)

        st.caption("Optional advanced fields (from your notebook)")
        model_name = st.text_input("TTS model_name (optional)", value="")  # e.g. gemini-2.5-pro-tts
        style_prompt = st.text_area("TTS prompt (optional)", value="", height=120)

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["1) Generate scripts", "2) Generate audio", "3) YouTube metadata"])

# ----------------------------
# Tab 1: Scripts
# ----------------------------
with tab1:
    st.subheader("Generate 2–3 minute scripts from course URLs")

    default_urls = ""

    left, right = st.columns([2, 1], gap="large")
    with left:
        urls_text = st.text_area(
            "One URL per line",
            value=default_urls,
            height=220,
            placeholder="https://example.com/course-a\nhttps://example.com/course-b",
        )
    with right:
        upload_csv = st.file_uploader("Or upload CSV with column 'link' or 'url'", type=["csv"])
        use_csv = st.toggle("Use uploaded CSV", value=False)

    def parse_urls() -> List[str]:
        if use_csv and upload_csv is not None:
            dfu = pd.read_csv(upload_csv)
            col = next((c for c in dfu.columns if c.lower() in ("link", "url")), None)
            if col is None:
                st.error("CSV must have a 'link' or 'url' column.")
                return []
            return dedupe_preserve_order([str(x) for x in dfu[col].dropna().tolist()])
        return dedupe_preserve_order(urls_text.splitlines())

    urls = parse_urls()
    st.write(f"URLs queued: **{len(urls)}**")

    run = st.button("Generate scripts", type="primary", use_container_width=True, disabled=(not gemini_api_key or not urls))

    if run:
        client = get_gemini_client(gemini_api_key)

        prog = st.progress(0.0)
        status = st.empty()

        def progress(i: int, total: int, label: str):
            status.write(f"[{i}/{total}] {label}")
            prog.progress(i / total)

        df, failed_df = run_script_generation(
            gemini_client=client,
            model=gemini_script_model,
            system_instruction=SYSTEM_INSTRUCTION_LONG,
            urls=urls,
            min_seconds_between_calls=min_seconds_scripts,
            progress=progress,
        )

        st.session_state["df_scripts"] = df
        st.session_state["failures"] = failed_df

        st.success(f"Done. Success: {len(df)} • Failed: {len(failed_df)}")

    if st.session_state["df_scripts"] is not None:
        df = st.session_state["df_scripts"]
        st.dataframe(df, use_container_width=True, height=420)

        csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("Download scripts CSV", data=csv_bytes, file_name="course_scripts_2_3min.csv", mime="text/csv")

    if st.session_state["failures"] is not None and len(st.session_state["failures"]) > 0:
        st.subheader("Failures")
        st.dataframe(st.session_state["failures"], use_container_width=True)

# ----------------------------
# Tab 2: Audio
# ----------------------------
with tab2:
    st.subheader("Generate MP3 audio from scripts (Google Cloud Text-to-Speech)")

    st.write("Input can be the scripts you generated in Tab 1, or a CSV you upload here.")
    upload_scripts_csv = st.file_uploader("Upload scripts CSV (title, link, script_2_3min)", type=["csv"], key="audio_csv")
    use_tab1 = st.toggle("Use Tab 1 results", value=True)

    df_in = None
    if use_tab1 and st.session_state["df_scripts"] is not None:
        df_in = st.session_state["df_scripts"]
    elif upload_scripts_csv is not None:
        df_in = pd.read_csv(upload_scripts_csv)

    if df_in is None:
        st.info("Provide scripts first (Tab 1) or upload a CSV.")
    else:
        st.dataframe(df_in.head(10), use_container_width=True)

        run_audio = st.button(
            "Generate MP3 files",
            type="primary",
            use_container_width=True,
            disabled=(not enable_tts or not voice_name),
        )

        if run_audio:
            try:
                tts_client = get_tts_client(
                    service_account_json=sa_json or None,
                    access_token=access_token or None,
                )
                if not (sa_json or access_token):
                    creds, proj = google.auth.default()
                    st.caption(f"ADC project: {proj} • quota_project: {getattr(creds, 'quota_project_id', None)}")
                else:
                    st.caption("Using provided TTS credentials (no ADC).")
            except Exception as e:
                st.error("Could not initialize TTS client. Check ADC setup.")
                st.exception(e)
                tts_client = None

            if tts_client:
                prog = st.progress(0.0)
                status = st.empty()

                def progress(i: int, total: int, label: str):
                    status.write(f"[{i}/{total}] {label}")
                    prog.progress(i / total)

                df_audio, blobs = run_tts_generation(
                    tts_client=tts_client,
                    df_scripts=df_in,
                    language_code=language_code,
                    voice_name=voice_name,
                    model_name=model_name or None,
                    prompt=style_prompt or None,
                    progress=progress,
                )

                st.session_state["df_audio"] = df_audio
                st.session_state["audio_blobs"] = blobs

                st.success(f"Generated {len(blobs)} MP3 files.")

        if st.session_state["df_audio"] is not None:
            df_audio = st.session_state["df_audio"]
            st.subheader("Audio CSV")
            st.dataframe(df_audio, use_container_width=True, height=420)

            csv_bytes = df_audio.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("Download audio CSV", data=csv_bytes, file_name="course_scripts_2_3min_with_audio.csv", mime="text/csv")

            if st.session_state["audio_blobs"]:
                # Build ZIP: mp3/ + csv
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("course_scripts_2_3min_with_audio.csv", csv_bytes)
                    for fname, blob in st.session_state["audio_blobs"]:
                        zf.writestr(f"mp3/{fname}", blob)
                zip_buf.seek(0)

                st.download_button(
                    "Download ZIP (CSV + MP3s)",
                    data=zip_buf.getvalue(),
                    file_name="scripts_with_audio.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

# ----------------------------
# Tab 3: YouTube metadata
# ----------------------------
with tab3:
    st.subheader("Generate YouTube metadata (needs audio to compute duration phrase)")

    st.write("Best flow: generate scripts → generate audio → generate YouTube metadata in one session.")
    use_audio_tab = st.toggle("Use Tab 2 results", value=True)

    df_audio = None
    blobs_by_name: Dict[str, bytes] = {}

    if use_audio_tab and st.session_state["df_audio"] is not None:
        df_audio = st.session_state["df_audio"]
        blobs_by_name = {fname: blob for fname, blob in st.session_state["audio_blobs"]}
    else:
        upload_audio_csv = st.file_uploader("Upload audio CSV (must include audio_2_3min_file)", type=["csv"], key="yt_csv")
        upload_zip = st.file_uploader("Upload ZIP of MP3s (mp3/<file>.mp3)", type=["zip"], key="yt_zip")
        if upload_audio_csv is not None:
            df_audio = pd.read_csv(upload_audio_csv)
        if upload_zip is not None:
            with zipfile.ZipFile(upload_zip) as zf:
                for n in zf.namelist():
                    if n.lower().endswith(".mp3"):
                        blobs_by_name[n.split("/")[-1]] = zf.read(n)

    if df_audio is None:
        st.info("Provide audio CSV + MP3s (Tab 2 or uploads).")
    else:
        st.dataframe(df_audio.head(10), use_container_width=True)

        # Pre-flight checks to explain why rows may be skipped
        required_cols = ["title", "link", "script_2_3min", "audio_2_3min_file"]
        missing_cols = [c for c in required_cols if c not in df_audio.columns]
        st.caption(f"MP3s available in session/ZIP: {len(blobs_by_name)}")
        if missing_cols:
            st.warning(f"Missing required columns: {', '.join(missing_cols)}")

        total_rows = len(df_audio)
        missing_fields = 0
        missing_mp3 = 0
        already_filled = 0
        ready_rows = 0

        for _, row in df_audio.iterrows():
            title = str(row.get("title", "")).strip()
            link = str(row.get("link", "")).strip()
            script = str(row.get("script_2_3min", "")).strip()
            audio_file = str(row.get("audio_2_3min_file", "")).strip()
            yt_desc = str(row.get("yt_description", "")).strip()

            if not title or not link or not script or not audio_file:
                missing_fields += 1
                continue
            if audio_file not in blobs_by_name:
                missing_mp3 += 1
                continue
            if yt_desc and yt_desc.lower() != "nan":
                already_filled += 1
                continue
            ready_rows += 1

        if missing_fields or missing_mp3 or already_filled:
            st.warning(
                "Some rows will be skipped. "
                f"Missing required fields: {missing_fields} • "
                f"MP3 not found: {missing_mp3} • "
                f"Already has metadata: {already_filled} • "
                f"Total rows: {total_rows}"
            )
        st.info(f"Rows ready for YouTube metadata generation: {ready_rows}")

        run_yt = st.button(
            "Generate YouTube metadata",
            type="primary",
            use_container_width=True,
            disabled=(not gemini_api_key),
        )

        if run_yt:
            client = get_gemini_client(gemini_api_key)

            prog = st.progress(0.0)
            status = st.empty()

            def progress(i: int, total: int, label: str):
                status.write(f"[{i}/{total}] {label}")
                prog.progress(i / total)

            df_yt = run_youtube_metadata(
                gemini_client=client,
                model=gemini_yt_model,
                system_instruction=SYSTEM_METADATA,
                df=df_audio,
                audio_blobs_by_name=blobs_by_name,
                min_seconds_between_calls=min_seconds_yt,
                progress=progress,
            )

            st.session_state["df_yt"] = df_yt
            # Post-run visibility: did anything get filled?
            filled = 0
            if df_yt is not None and "yt_description" in df_yt.columns:
                filled = df_yt["yt_description"].fillna("").astype(str).str.strip().ne("").sum()
            errors = 0
            if df_yt is not None and "yt_error" in df_yt.columns:
                errors = df_yt["yt_error"].fillna("").astype(str).str.strip().ne("").sum()
            if filled == 0:
                st.warning("No YouTube metadata was generated. Check the warning above for skipped rows.")
                if errors:
                    st.error(f"Errors during generation: {errors}. See yt_error column.")
            else:
                if errors:
                    st.warning(f"Generated metadata with {errors} errors. See yt_error column.")
                st.success("Done.")

    if st.session_state["df_yt"] is not None:
        df_yt = st.session_state["df_yt"]
        st.subheader("YouTube metadata output")
        st.dataframe(df_yt, use_container_width=True, height=420)

        csv_bytes = df_yt.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("Download CSV with YouTube metadata", data=csv_bytes, file_name="course_scripts_with_audio_and_yt.csv", mime="text/csv")
