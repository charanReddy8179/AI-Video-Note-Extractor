import streamlit as st
from moviepy import VideoFileClip
import whisper
import os
import yt_dlp
import time
from fpdf import FPDF
from transformers import pipeline


# ---------------- UI ----------------

st.title("🎥 AI Video Note Extractor")

st.write(
    "Upload a video or paste a YouTube link to generate notes and download them as PDF."
)

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "mov", "avi"]
)

youtube_link = st.text_input(
    "Paste YouTube Video URL"
)

video_path = None


# ---------------- YOUTUBE DOWNLOAD ----------------

def download_youtube_video(url):

    filename = f"youtube_video_{int(time.time())}.mp4"

    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": filename
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return filename


# ---------------- PDF GENERATION ----------------

def create_pdf(transcript, notes, revision):

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=12)

    transcript = transcript.encode(
        "latin-1",
        "replace"
    ).decode("latin-1")

    # Header
    pdf.cell(0, 10, "Video Notes")
    pdf.ln(12)

    # Transcript Section
    pdf.cell(0, 10, "Transcript")
    pdf.ln(10)
    pdf.multi_cell(0, 10, transcript)
    pdf.ln(10)

    # Detailed Notes Section
    pdf.cell(0, 10, "Detailed Notes")
    pdf.ln(10)

    for note in notes:

        note = note.encode(
            "latin-1",
            "replace"
        ).decode("latin-1")

        pdf.multi_cell(0, 10, note)
        pdf.ln(4) # small gap between bullet points

    pdf.ln(6)

    # Quick Revision Section
    pdf.cell(0, 10, "Quick Revision")
    pdf.ln(10)

    for point in revision:

        point = point.encode(
            "latin-1",
            "replace"
        ).decode("latin-1")

        pdf.multi_cell(0, 10, point)
        pdf.ln(4)

    pdf.output("notes.pdf")


# ---------------- LOAD MODELS ----------------

@st.cache_resource
def load_models():

    whisper_model = whisper.load_model("tiny")  # 'tiny' uses ~70MB RAM vs 500MB for 'base'

    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-6-6"  # '6-6' uses half the RAM of '12-6'
    )

    return whisper_model, summarizer


whisper_model, summarizer = load_models()


# ---------------- CHUNKED SUMMARIZATION ----------------

def chunk_and_summarize(transcript, summarizer, chunk_size=1800, overlap=100):
    """
    Splits a long transcript into overlapping chunks,
    summarizes each chunk separately, then returns
    ALL chunk summaries joined together.

    KEY FIX: We do NOT do a final truncation pass.
    Instead, we return all chunk summaries joined so
    that EVERY part of the video appears in the notes.

    Args:
        transcript  : full text string (any length)
        summarizer  : loaded HuggingFace pipeline
        chunk_size  : max characters per chunk (safe limit for DistilBART)
        overlap     : characters shared between consecutive chunks
                      (preserves sentence context at boundaries)

    Returns:
        summarized_text : all chunk summaries joined (covers full video)
    """

    # ── SHORT TRANSCRIPT: no chunking needed ──
    if len(transcript) <= chunk_size:
        result = summarizer(
            transcript,
            max_length=150,
            min_length=50,
            do_sample=False
        )
        return result[0]["summary_text"]

    # ── LONG TRANSCRIPT: split into chunks ──

    # STEP 1 — Build chunks with overlap
    chunks = []
    start = 0

    while start < len(transcript):
        end = start + chunk_size
        chunk = transcript[start:end]
        chunks.append(chunk)
        # Move forward by chunk_size MINUS overlap
        # so next chunk starts slightly before this one ended
        # This prevents sentences from being cut at boundaries
        start += (chunk_size - overlap)

    total_chunks = len(chunks)

    st.write(f"📄 Long transcript detected — {len(transcript)} characters")
    st.write(f"🔀 Splitting into {total_chunks} parts for complete summarization...")

    # STEP 2 — Summarize each chunk individually
    chunk_summaries = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, chunk in enumerate(chunks):

        status_text.write(
            f"⏳ Summarizing part {i + 1} of {total_chunks}..."
        )

        result = summarizer(
            chunk,
            max_length=120,
            min_length=30,
            do_sample=False
        )

        chunk_summaries.append(result[0]["summary_text"])

        # Update progress bar (value must be between 0.0 and 1.0)
        progress_bar.progress((i + 1) / total_chunks)

    progress_bar.empty()
    status_text.empty()

    # STEP 3 — Join ALL chunk summaries with a period separator
    # No truncation here — every chunk's summary is kept
    # This ensures the full video is represented in the notes
    combined_summary = ". ".join(chunk_summaries)

    return combined_summary


# ---------------- HANDLE INPUT ----------------

if uploaded_file is not None:

    with open("input_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    video_path = "input_video.mp4"

    st.success("Video uploaded successfully")

elif youtube_link:

    st.write("Downloading YouTube video...")

    try:

        video_path = download_youtube_video(
            youtube_link
        )

        st.success(
            "YouTube video downloaded successfully"
        )

    except Exception as e:

        st.error(
            f"YouTube download failed: {e}"
        )


# ---------------- PROCESS VIDEO ----------------

if video_path:

    try:

        st.write(
            f"Processing: {video_path}"
        )

        video = VideoFileClip(video_path)

        if video.audio is None:

            st.error(
                "Video does not contain audio"
            )

        else:

            if os.path.exists("audio.mp3"):
                os.remove("audio.mp3")

            video.audio.write_audiofile(
                "audio.mp3"
            )

            st.success(
                "Audio extracted successfully"
            )

            st.write(
                "Transcribing using Whisper..."
            )

            result = whisper_model.transcribe(
                "audio.mp3",
                fp16=False
            )

            transcript = result["text"]

            if len(transcript.strip()) == 0:

                st.error(
                    "Transcript is empty"
                )

            else:

                st.success(
                    "Transcription completed"
                )

                st.write(
                    "### Transcript Preview"
                )

                st.write(
                    transcript[:500]
                )

                st.write(
                    f"Transcript Length: {len(transcript)} characters"
                )

                # ── CHUNKED SUMMARIZATION (handles any length) ──
                st.write("Generating AI Notes...")

                summarized_text = chunk_and_summarize(
                    transcript,
                    summarizer
                )

                # ---------------- NOTES ----------------

                sentences = summarized_text.replace(
                    "\n",
                    "."
                ).split(".")

                notes_list = []

                for sentence in sentences:

                    sentence = sentence.strip()

                    if len(sentence) > 15:

                        notes_list.append(
                            "- " + sentence
                        )

                if len(notes_list) == 0:

                    notes_list.append(
                        "- Summary generation completed."
                    )

                st.success(
                    "Notes generated successfully"
                )

                st.write(
                    "### Detailed Notes"
                )

                for note in notes_list:

                    st.write(note)

                revision_points = notes_list[:6]

                st.write(
                    "### Quick Revision"
                )

                for point in revision_points:

                    st.write(point)

                # ---------------- PDF ----------------

                create_pdf(
                    transcript,
                    notes_list,
                    revision_points
                )

                with open(
                    "notes.pdf",
                    "rb"
                ) as file:

                    st.download_button(
                        label="📥 Download PDF",
                        data=file,
                        file_name="video_notes.pdf",
                        mime="application/pdf"
                    )

    except Exception as e:

        st.error(
            f"Processing failed: {e}"
        )
