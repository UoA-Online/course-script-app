import io
import zipfile
from typing import Dict, List

import pandas as pd
import streamlit as st

from src.gemini_client import get_client as get_gemini_client
from src.pipeline import run_script_generation, run_tts_generation, run_youtube_metadata
from src.prompts import SYSTEM_INSTRUCTION_LONG, SYSTEM_METADATA
from src.tts_client import DEFAULT_MAX_CHUNK_BYTES, DEFAULT_SAMPLE_RATE_HERTZ, get_tts_client
from src.utils import dedupe_preserve_order

DEFAULT_TTS_STYLE_PROMPT = (
    "Speak in a natural UK English accent. "
    "Sound warm, conversational, and human like a university course narrator. "
    "Use varied intonation, light emphasis, and natural pauses. "
    "Avoid sounding like a robot or reading an advert."
)


def resolve_service_account_value(uploaded_file, text_value: str) -> str:
    if uploaded_file is not None:
        return uploaded_file.getvalue().decode("utf-8")
    return (text_value or "").strip()


def load_audio_blobs_from_zip(uploaded_zip) -> Dict[str, bytes]:
    blobs_by_name: Dict[str, bytes] = {}
    with zipfile.ZipFile(uploaded_zip) as zf:
        for name in zf.namelist():
            lower_name = name.lower()
            if lower_name.endswith(".wav") or lower_name.endswith(".mp3"):
                blobs_by_name[name.split("/")[-1]] = zf.read(name)
    return blobs_by_name


st.set_page_config(page_title="Course Script + Audio + YouTube Metadata", layout="wide")
st.title("Course Script Generator (Gemini) + Audio (Gemini TTS) + YouTube Metadata")

if "df_scripts" not in st.session_state:
    st.session_state["df_scripts"] = None
if "df_audio" not in st.session_state:
    st.session_state["df_audio"] = None
if "audio_blobs" not in st.session_state:
    st.session_state["audio_blobs"] = []
if "df_yt" not in st.session_state:
    st.session_state["df_yt"] = None
if "failures" not in st.session_state:
    st.session_state["failures"] = None

with st.sidebar:
    st.header("Credentials")

    gemini_api_key = st.text_input(
        "Gemini API key (GEMINI_API_KEY)",
        type="password",
        value=st.secrets.get("GEMINI_API_KEY", ""),
        help="Use Streamlit secrets in deployment; do not hardcode keys.",
    )

    st.header("Gemini models")
    gemini_script_model = st.text_input("Script model", value="gemini-2.5-pro")
    gemini_yt_model = st.text_input("YouTube metadata model", value="gemini-2.0-flash")

    st.header("Rate limits")
    min_seconds_scripts = st.slider("Seconds between script calls", 0.0, 10.0, 3.0, 0.5)
    min_seconds_tts = st.slider("Seconds between TTS chunk calls", 0.0, 10.0, 0.5, 0.5)
    min_seconds_yt = st.slider("Seconds between metadata calls", 0.0, 10.0, 1.5, 0.5)

    st.header("Gemini TTS")
    st.caption("Audio generation now uses the demo service-account flow and exports WAV files.")
    language_code = st.text_input("Language code", value="en-GB")
    voice_name = st.text_input("Voice name", value="Vindemiatrix")
    model_name = st.text_input("TTS model", value="gemini-2.5-pro-tts")
    style_prompt = st.text_area("Style prompt", value=DEFAULT_TTS_STYLE_PROMPT, height=140)

    with st.expander("Service account", expanded=False):
        uploaded_service_account = st.file_uploader("Upload service account JSON", type=["json"], key="tts_service_account_file")
        service_account_text = st.text_area(
            "Or paste service account JSON or a local file path",
            value=st.secrets.get("GCP_TTS_SERVICE_ACCOUNT_JSON", ""),
            height=160,
            placeholder='{"type":"service_account",...}',
        )
        if uploaded_service_account is not None:
            st.caption(f"Using uploaded file: {uploaded_service_account.name}")

    with st.expander("Advanced audio settings", expanded=False):
        sample_rate_hertz = st.number_input(
            "Sample rate (Hz)",
            min_value=8000,
            max_value=48000,
            value=DEFAULT_SAMPLE_RATE_HERTZ,
            step=1000,
        )
        max_chunk_bytes = st.number_input(
            "Max bytes per TTS chunk",
            min_value=200,
            max_value=4000,
            value=DEFAULT_MAX_CHUNK_BYTES,
            step=100,
            help="Keeps each request under the demo TTS chunk size limit.",
        )

service_account_value = resolve_service_account_value(uploaded_service_account, service_account_text)

tab1, tab2, tab3 = st.tabs(["1) Generate scripts", "2) Generate audio", "3) YouTube metadata"])

with tab1:
    st.subheader("Generate 2-3 minute scripts from course URLs")

    left, right = st.columns([2, 1], gap="large")
    with left:
        urls_text = st.text_area(
            "One URL per line",
            value="",
            height=220,
            placeholder="https://example.com/course-a\nhttps://example.com/course-b",
        )
    with right:
        upload_csv = st.file_uploader("Or upload CSV with column 'link' or 'url'", type=["csv"])
        use_csv = st.toggle("Use uploaded CSV", value=False)

    def parse_urls() -> List[str]:
        if use_csv and upload_csv is not None:
            df_uploaded = pd.read_csv(upload_csv)
            column = next((col for col in df_uploaded.columns if col.lower() in ("link", "url")), None)
            if column is None:
                st.error("CSV must have a 'link' or 'url' column.")
                return []
            return dedupe_preserve_order([str(item) for item in df_uploaded[column].dropna().tolist()])
        return dedupe_preserve_order(urls_text.splitlines())

    urls = parse_urls()
    st.write(f"URLs queued: **{len(urls)}**")

    run_scripts = st.button(
        "Generate scripts",
        type="primary",
        use_container_width=True,
        disabled=(not gemini_api_key or not urls),
    )

    if run_scripts:
        client = get_gemini_client(gemini_api_key)
        prog = st.progress(0.0)
        status = st.empty()

        def progress(i: int, total: int, label: str) -> None:
            status.write(f"[{i}/{total}] {label}")
            prog.progress(i / total)

        df_scripts, failed_df = run_script_generation(
            gemini_client=client,
            model=gemini_script_model,
            system_instruction=SYSTEM_INSTRUCTION_LONG,
            urls=urls,
            min_seconds_between_calls=min_seconds_scripts,
            progress=progress,
        )

        st.session_state["df_scripts"] = df_scripts
        st.session_state["failures"] = failed_df
        st.success(f"Done. Success: {len(df_scripts)} • Failed: {len(failed_df)}")

    if st.session_state["df_scripts"] is not None:
        df_scripts = st.session_state["df_scripts"]
        st.dataframe(df_scripts, use_container_width=True, height=420)

        csv_bytes = df_scripts.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "Download scripts CSV",
            data=csv_bytes,
            file_name="course_scripts_2_3min.csv",
            mime="text/csv",
        )

    if st.session_state["failures"] is not None and len(st.session_state["failures"]) > 0:
        st.subheader("Failures")
        st.dataframe(st.session_state["failures"], use_container_width=True)

with tab2:
    st.subheader("Generate WAV audio from scripts (Gemini TTS service-account flow)")
    st.write("Input can be the scripts you generated in Tab 1, or a CSV you upload here.")

    upload_scripts_csv = st.file_uploader("Upload scripts CSV (title, link, script_2_3min)", type=["csv"], key="audio_csv")
    use_tab1 = st.toggle("Use Tab 1 results", value=True)

    df_in = None
    if use_tab1 and st.session_state["df_scripts"] is not None:
        df_in = st.session_state["df_scripts"]
    elif upload_scripts_csv is not None:
        df_in = pd.read_csv(upload_scripts_csv)

    if not service_account_value:
        st.warning("Add a service account JSON in the sidebar to enable audio generation.")

    if df_in is None:
        st.info("Provide scripts first (Tab 1) or upload a CSV.")
    else:
        st.dataframe(df_in.head(10), use_container_width=True)

        run_audio = st.button(
            "Generate WAV files",
            type="primary",
            use_container_width=True,
            disabled=(not service_account_value or not voice_name or not model_name),
        )

        if run_audio:
            try:
                tts_client = get_tts_client(service_account_json=service_account_value)
                st.caption(f"Using service account project: {tts_client.project_id}")
            except Exception as exc:
                st.error("Could not initialize Gemini TTS. Check the service account JSON.")
                st.exception(exc)
                tts_client = None

            if tts_client:
                prog = st.progress(0.0)
                status = st.empty()

                def progress(i: int, total: int, label: str) -> None:
                    status.write(f"[{i}/{total}] {label}")
                    prog.progress(i / total)

                df_audio, blobs = run_tts_generation(
                    tts_client=tts_client,
                    df_scripts=df_in,
                    language_code=language_code,
                    voice_name=voice_name,
                    model_name=model_name,
                    prompt=style_prompt or None,
                    sample_rate_hertz=int(sample_rate_hertz),
                    max_chunk_bytes=int(max_chunk_bytes),
                    min_seconds_between_calls=min_seconds_tts,
                    progress=progress,
                )

                st.session_state["df_audio"] = df_audio
                st.session_state["audio_blobs"] = blobs
                st.session_state["df_yt"] = None
                st.success(f"Generated {len(blobs)} WAV files.")

        if st.session_state["df_audio"] is not None:
            df_audio = st.session_state["df_audio"]
            st.subheader("Audio CSV")
            st.dataframe(df_audio, use_container_width=True, height=420)

            csv_bytes = df_audio.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "Download audio CSV",
                data=csv_bytes,
                file_name="course_scripts_2_3min_with_audio.csv",
                mime="text/csv",
            )

            if st.session_state["audio_blobs"]:
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("course_scripts_2_3min_with_audio.csv", csv_bytes)
                    for fname, blob in st.session_state["audio_blobs"]:
                        zf.writestr(f"audio/{fname}", blob)
                zip_buf.seek(0)

                st.download_button(
                    "Download ZIP (CSV + WAVs)",
                    data=zip_buf.getvalue(),
                    file_name="scripts_with_audio.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

with tab3:
    st.subheader("Generate YouTube metadata (needs audio to compute duration phrase)")
    st.write("Best flow: generate scripts -> generate audio -> generate YouTube metadata in one session.")

    use_audio_tab = st.toggle("Use Tab 2 results", value=True)
    df_audio = None
    blobs_by_name: Dict[str, bytes] = {}

    if use_audio_tab and st.session_state["df_audio"] is not None:
        df_audio = st.session_state["df_audio"]
        blobs_by_name = {fname: blob for fname, blob in st.session_state["audio_blobs"]}
    else:
        upload_audio_csv = st.file_uploader("Upload audio CSV (must include audio_2_3min_file)", type=["csv"], key="yt_csv")
        upload_zip = st.file_uploader("Upload ZIP of audio files", type=["zip"], key="yt_zip")
        if upload_audio_csv is not None:
            df_audio = pd.read_csv(upload_audio_csv)
        if upload_zip is not None:
            blobs_by_name = load_audio_blobs_from_zip(upload_zip)

    if df_audio is None:
        st.info("Provide audio CSV + audio files (Tab 2 or uploads).")
    else:
        st.dataframe(df_audio.head(10), use_container_width=True)

        required_cols = ["title", "link", "script_2_3min", "audio_2_3min_file"]
        missing_cols = [col for col in required_cols if col not in df_audio.columns]
        st.caption(f"Audio files available in session/ZIP: {len(blobs_by_name)}")
        if missing_cols:
            st.warning(f"Missing required columns: {', '.join(missing_cols)}")

        total_rows = len(df_audio)
        missing_fields = 0
        missing_audio = 0
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
                missing_audio += 1
                continue
            if yt_desc and yt_desc.lower() != "nan":
                already_filled += 1
                continue
            ready_rows += 1

        if missing_fields or missing_audio or already_filled:
            st.warning(
                "Some rows will be skipped. "
                f"Missing required fields: {missing_fields} • "
                f"Audio file not found: {missing_audio} • "
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

            def progress(i: int, total: int, label: str) -> None:
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
        st.download_button(
            "Download CSV with YouTube metadata",
            data=csv_bytes,
            file_name="course_scripts_with_audio_and_yt.csv",
            mime="text/csv",
        )
