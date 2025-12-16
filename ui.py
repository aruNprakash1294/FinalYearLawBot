import streamlit as st
from main import transcribe_tamil_audio, find_similar_cases

st.set_page_config(page_title="Tamil Legal Assistance System", layout="centered")

st.title("⚖️ Tamil Legal Assistance System")
st.write(
    "Ask your legal query using **voice recording**, **audio upload**, or **text input** "
    "related to police misuse, intimidation, or abuse of power."
)

# -----------------------------
# Input Section
# -----------------------------

st.subheader("🎤 Record Tamil Audio")
recorded_audio = st.audio_input("Click to record")

st.subheader("📁 Upload Audio File")
uploaded_audio = st.file_uploader(
    "Upload Tamil Audio (.mp3 / .wav / .m4a)",
    type=["mp3", "wav", "m4a"]
)

st.subheader("⌨️ Enter Text (Tamil or English)")
text_input = st.text_area(
    "Type your legal query",
    placeholder="என்னை காவல்துறை மிரட்டியது, என்ன செய்ய வேண்டும்?"
)

audio_path = None
query_text = None

# -----------------------------
# Priority Handling
# -----------------------------

if recorded_audio is not None:
    audio_path = "temp_recorded_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(recorded_audio.getbuffer())
    st.success("Recorded audio captured")

elif uploaded_audio is not None:
    audio_path = "temp_uploaded_audio.m4a"
    with open(audio_path, "wb") as f:
        f.write(uploaded_audio.read())
    st.success("Uploaded audio saved")

elif text_input.strip():
    query_text = text_input
    st.success("Text query received")

# -----------------------------
# Processing
# -----------------------------

if st.button("Process Query"):
    if audio_path:
        with st.spinner("Transcribing audio using Whisper..."):
            query_text = transcribe_tamil_audio(audio_path)

    if not query_text:
        st.error("Please provide audio or text input.")
        st.stop()

    st.subheader("🗣 User Query")
    st.write(query_text)

    # Temporary English query (later auto-translate)
    english_query = "Police intimidation and misuse of power against civilians"

    with st.spinner("Finding similar legal cases..."):
        similar_cases = find_similar_cases(english_query, top_k=3)

    st.subheader("📚 Similar Legal Cases & Outcomes")
    for i, case in enumerate(similar_cases, 1):
        st.markdown(f"**Case {i}:**")
        st.write(case)
        st.markdown("---")
