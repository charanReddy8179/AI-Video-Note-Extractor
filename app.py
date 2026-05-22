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

    pdf.cell(200, 10, "Video Notes", ln=True)

    pdf.cell(200, 10, "Transcript", ln=True)

    pdf.multi_cell(0, 10, transcript)

    pdf.cell(200, 10, "Detailed Notes", ln=True)

    for note in notes:

        note = note.encode(
            "latin-1",
            "replace"
        ).decode("latin-1")

        pdf.multi_cell(0, 10, note)

    pdf.cell(200, 10, "Quick Revision", ln=True)

    for point in revision:

        point = point.encode(
            "latin-1",
            "replace"
        ).decode("latin-1")

        pdf.multi_cell(0, 10, point)

    pdf.output("notes.pdf")


# ---------------- LOAD MODELS ----------------

@st.cache_resource
def load_models():

    whisper_model = whisper.load_model("base")

    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

    return whisper_model, summarizer


whisper_model, summarizer = load_models()


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
                    f"Transcript Length: {len(transcript)}"
                )

                # ---------------- SAFE SUMMARY INPUT ----------------

                MAX_CHARS = 1800

                if len(transcript) > MAX_CHARS:

                    short_transcript = transcript[
                        :MAX_CHARS
                    ]

                else:

                    short_transcript = transcript

                st.write(
                    "Generating AI Notes..."
                )

                summary = summarizer(
                    short_transcript,
                    max_length=150,
                    min_length=50,
                    do_sample=False
                )

                summarized_text = summary[
                    0
                ][
                    "summary_text"
                ]

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
