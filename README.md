# 🎤 Nigerian Languages Speech Transcription

This Gradio application provides a user-friendly interface for transcribing spoken audio in three major Nigerian languages: **Yoruba**, **Hausa**, and **Igbo**. It leverages specialized Whisper and Wav2Vec2 models to convert speech to text, making it easier to process and analyze audio content in these languages.

---

## ✨ Features

* **Multi-language Support:** Transcribe audio in Yoruba, Hausa, and Igbo.
* **Flexible Input:** Upload audio files (WAV, MP3, etc.) or record directly from your microphone.
* **Automatic Chunking:** Handles long audio files by automatically splitting and processing them in chunks for efficient transcription.
* **Real-time Feedback:** Provides transcription results quickly.
* **Model Information:** Displays details about the underlying models used for each language.
* **User-Friendly Interface:** Built with Gradio for an intuitive and interactive experience.

---

## 🛠️ Installation

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### Steps

1.  **Clone the repository (or save `app.py` and `requirements.txt`):**

    ```bash
    git clone https://github.com/samolubukun/Nigerian-Languages-WAZOBIA-Speech-Transcription.git
    cd Nigerian-Languages-WAZOBIA-Speech-Transcription
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    The `requirements.txt` file contains all the necessary Python packages.

    ```bash
    pip install -r requirements.txt
    ```

    Expected content of `requirements.txt`:

    ```
    gradio
    torch
    transformers
    librosa
    numpy
    ```

---

## 🚀 Usage

1.  **Run the application:**

    ```bash
    python app.py
    ```

2.  **Access the interface:**

    Once the application starts, it will provide a local URL (e.g., `http://127.0.0.1:7860`). If `share=True` is enabled in `app.py`, it will also provide a public Gradio Share link. Open this URL in your web browser.

3.  **How to Use the Interface:**

    * **Select Language:** Choose your desired language (Yoruba, Hausa, or Igbo) from the "Select Language" dropdown. The "Model Information" box will update to show details about the selected model.
    * **Input Audio:**
        * **Upload File:** Click on the "Upload File" tab and drag-and-drop or browse to select an audio file (e.g., `.wav`, `.mp3`).
        * **Record Speech:** Click on the "Record Speech" tab and use the microphone icon to record your voice directly.
    * **Transcribe:** Click the "🎯 Transcribe Audio" button to start the transcription process.
    * **View Result:** The transcribed text will appear in the "Transcription Result" box. You can use the copy button to easily copy the text.

---

## Screenshot

![png (4)](https://github.com/user-attachments/assets/58e9ebea-d801-4b7d-812c-a41d6e73d0c0)


## 💡 Notes

* **First-time Model Loading:** The first time you select a language, the corresponding model will be downloaded. This might take a few minutes depending on your internet connection. Subsequent uses of the same language will be faster as the model will be cached.
* **Audio Quality:** For best results, ensure your audio recordings are clear with minimal background noise.
* **Long Audio Handling:** Audio files longer than approximately 25 seconds will be automatically processed in smaller chunks to prevent memory issues and improve performance.

---

## 🧠 Models Used

The application utilizes different models specialized for each language:

* **Yoruba:** `DereAbdulhameed/Whisper-Yoruba` (Fine-tuned Whisper model)
* **Hausa:** `Baghdad99/saad-speech-recognition-hausa-audio-to-text` (Fine-tuned Whisper model)
* **Igbo:** `AstralZander/igbo_ASR` (Fine-tuned Wav2Vec2-XLS-R model)

---


## 🙏 Acknowledgements

* The creators of the specialized ASR models for Yoruba, Hausa, and Igbo.
* Hugging Face Transformers library
* Gradio for the intuitive UI
