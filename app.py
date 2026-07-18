import os
import re
import tempfile
import time
from pathlib import Path

import streamlit as st
import whisper
import yt_dlp
from fpdf import FPDF

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip


st.set_page_config(page_title="AI Video Note Extractor", page_icon="V")
OUTPUT_DIR = Path(tempfile.gettempdir()) / "ai_video_note_extractor"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_pdf_text(text):
    return text.encode("latin-1", "replace").decode("latin-1")


def safe_filename(name):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_") or "video"


def save_uploaded_video(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    filename = OUTPUT_DIR / f"{safe_filename(Path(uploaded_file.name).stem)}_{int(time.time())}{suffix}"
    with open(filename, "wb") as output_file:
        output_file.write(uploaded_file.getbuffer())
    return str(filename)


def download_youtube_video(url):
    output_template = str(OUTPUT_DIR / f"youtube_{int(time.time())}.%(ext)s")
    options = {
        "format": "best[ext=mp4]/best",
        "outtmpl": output_template,
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=True)
        downloaded_path = downloader.prepare_filename(info)
    if not Path(downloaded_path).exists():
        raise FileNotFoundError("The YouTube video was not downloaded.")
    return downloaded_path


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


def make_notes(transcript, maximum_notes=10):
    """Build useful extractive notes without Transformers summarization pipelines."""
    sentences = [
        " ".join(sentence.split()).strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", transcript)
    ]
    sentences = [sentence for sentence in sentences if len(sentence.split()) >= 5]

    if not sentences:
        return ["- No clear spoken sentences were detected in this video."]

    stop_words = {
        "about", "after", "again", "also", "because", "being", "between", "could",
        "every", "from", "have", "into", "just", "more", "most", "only", "other",
        "over", "really", "should", "some", "than", "that", "their", "there", "these",
        "they", "this", "through", "under", "very", "what", "when", "which", "while",
        "with", "would", "your",
    }
    counts = {}
    for word in re.findall(r"[a-zA-Z]{3,}", transcript.lower()):
        if word not in stop_words:
            counts[word] = counts.get(word, 0) + 1

    scored = []
    for index, sentence in enumerate(sentences):
        words = re.findall(r"[a-zA-Z]{3,}", sentence.lower())
        if len(words) > 45:
            continue
        score = sum(counts.get(word, 0) for word in words) / max(len(words), 1)
        scored.append((score, index, sentence))

    selected = sorted(scored, reverse=True)[:maximum_notes]
    selected.sort(key=lambda item: item[1])
    return [f"- {sentence}" for _, _, sentence in selected]


def create_pdf(transcript, notes, output_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Video Notes", ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Key Notes", ln=True)
    pdf.set_font("Arial", size=11)
    for note in notes:
        pdf.multi_cell(0, 7, safe_pdf_text(note))

    pdf.ln(3)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Full Transcript", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, safe_pdf_text(transcript))
    pdf.output(output_path)


st.title("AI Video Note Extractor")
st.write("Upload a video or paste a YouTube link to generate notes and download them as a PDF.")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv", "mpeg"])
youtube_url = st.text_input("Paste YouTube Video URL")

try:
    whisper_model = load_whisper_model()
except Exception as error:
    st.error(f"Whisper model loading failed: {error}")
    st.stop()

video_path = None
if uploaded_file is not None:
    try:
        video_path = save_uploaded_video(uploaded_file)
        st.success("Video uploaded successfully.")
    except Exception as error:
        st.error(f"Could not save the uploaded video: {error}")
        st.stop()
elif youtube_url.strip():
    try:
        with st.spinner("Downloading YouTube video..."):
            video_path = download_youtube_video(youtube_url.strip())
        st.success("YouTube video downloaded successfully.")
    except Exception as error:
        st.error(f"YouTube download failed: {error}")
        st.stop()

if video_path:
    audio_path = str(OUTPUT_DIR / f"audio_{int(time.time())}.mp3")
    pdf_path = str(OUTPUT_DIR / f"video_notes_{int(time.time())}.pdf")

    try:
        with st.spinner("Extracting audio from video..."):
            with VideoFileClip(video_path) as video:
                if video.audio is None:
                    st.error("This video has no audio track.")
                    st.stop()
                video.audio.write_audiofile(audio_path, logger=None)

        with st.spinner("Transcribing the audio with Whisper..."):
            transcription = whisper_model.transcribe(audio_path, fp16=False)
        transcript = transcription.get("text", "").strip()

        if not transcript:
            st.error("No speech was detected in the video.")
            st.stop()

        notes = make_notes(transcript)
        create_pdf(transcript, notes, pdf_path)

        st.success("Notes generated successfully.")
        st.subheader("Key Notes")
        for note in notes:
            st.write(note)

        st.subheader("Transcript Preview")
        st.write(transcript[:1500])

        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="Download PDF",
                data=pdf_file.read(),
                file_name="video_notes.pdf",
                mime="application/pdf",
            )
    except Exception as error:
        st.error(f"Video processing failed: {error}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
