# app.py
import gradio as gr
import torch
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Wav2Vec2Processor, Wav2Vec2ForCTC
)
import librosa
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class NigerianWhisperTranscriber:
    def __init__(self):
        self.models = {}
        self.processors = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model configurations with their architectures
        self.model_configs = {
            "Yoruba": {
                "model_name": "DereAbdulhameed/Whisper-Yoruba",
                "architecture": "whisper"
            },
            "Hausa": {
                "model_name": "Baghdad99/saad-speech-recognition-hausa-audio-to-text",
                "architecture": "whisper"
            },
            "Igbo": {
                "model_name": "AstralZander/igbo_ASR",
                "architecture": "wav2vec2"
            }
        }

        print(f"Using device: {self.device}")

    def load_model(self, language):
        """Load model and processor for specific language"""
        if language not in self.models:
            try:
                print(f"Loading {language} model...")
                config = self.model_configs[language]
                model_name = config["model_name"]
                architecture = config["architecture"]

                if architecture == "whisper":
                    # Load Whisper model
                    processor = WhisperProcessor.from_pretrained(model_name)
                    model = WhisperForConditionalGeneration.from_pretrained(model_name)
                    model = model.to(self.device)

                elif architecture == "wav2vec2":
                    # Load Wav2Vec2 model
                    processor = Wav2Vec2Processor.from_pretrained(model_name)
                    model = Wav2Vec2ForCTC.from_pretrained(model_name)
                    model = model.to(self.device)

                self.processors[language] = processor
                self.models[language] = model

                print(f"{language} model loaded successfully!")
                return True
            except Exception as e:
                print(f"Error loading {language} model: {str(e)}")
                return False
        return True

    def preprocess_audio(self, audio_path):
        """Preprocess audio file for Whisper"""
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=16000)

            # Ensure audio is not empty
            if len(audio) == 0:
                raise ValueError("Audio file is empty")

            # Normalize audio
            audio = audio.astype(np.float32)

            return audio
        except Exception as e:
            raise ValueError(f"Error processing audio: {str(e)}")

    def chunk_audio(self, audio, chunk_length=25):
        """Split audio into chunks for processing longer recordings"""
        sample_rate = 16000
        chunk_samples = chunk_length * sample_rate

        chunks = []
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) > sample_rate:  # Only process chunks longer than 1 second
                chunks.append(chunk)

        return chunks

    def transcribe_chunk(self, audio_chunk, language):
        """Transcribe a single audio chunk"""
        processor = self.processors[language]
        model = self.models[language]
        config = self.model_configs[language]

        if config["architecture"] == "whisper":
            # Whisper processing
            inputs = processor(
                audio_chunk,
                sampling_rate=16000,
                return_tensors="pt"
            )

            input_features = inputs.input_features.to(self.device)

            # Create attention mask if available
            attention_mask = None
            if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None:
                attention_mask = inputs.attention_mask.to(self.device)

            # Generate transcription
            with torch.no_grad():
                if attention_mask is not None:
                    predicted_ids = model.generate(
                        input_features,
                        attention_mask=attention_mask,
                        max_new_tokens=400,
                        num_beams=5,
                        temperature=0.0,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                else:
                    predicted_ids = model.generate(
                        input_features,
                        max_new_tokens=400,
                        num_beams=5,
                        temperature=0.0,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )

            # Decode transcription
            transcription = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            return transcription.strip()

        elif config["architecture"] == "wav2vec2":
            # Wav2Vec2 processing
            inputs = processor(
                audio_chunk,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )

            input_values = inputs.input_values.to(self.device)

            # Generate transcription
            with torch.no_grad():
                logits = model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)

            # Decode transcription for Wav2Vec2
            # The key is to use `skip_special_tokens=True` here as well,
            # and potentially handle any remaining [PAD] explicitly if the tokenizer
            # doesn't completely remove them with that flag.
            transcription = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True  # Ensure special tokens are skipped
            )[0]

            # Additional clean-up for Wav2Vec2 specific models if skip_special_tokens isn't enough
            # Some Wav2Vec2 tokenizers might represent padding characters differently or
            # not fully remove them with skip_special_tokens=True depending on how they were trained.
            # We can perform an explicit string replacement as a fallback.
            transcription = transcription.replace("[PAD]", "").strip()
            transcription = " ".join(transcription.split())  # To remove extra spaces

            return transcription.strip()

    def transcribe(self, audio_path, language):
        """Transcribe audio file in specified language"""
        try:
            # Load model if not already loaded
            if not self.load_model(language):
                return f"Error: Could not load {language} model"

            # Preprocess audio
            audio = self.preprocess_audio(audio_path)

            # Check audio length (25 seconds = 400,000 samples at 16kHz)
            if len(audio) > 400000:  # If longer than 25 seconds
                # Process in chunks
                chunks = self.chunk_audio(audio, chunk_length=25)
                transcriptions = []

                for i, chunk in enumerate(chunks):
                    print(f"Processing chunk {i+1}/{len(chunks)}")

                    # Transcribe chunk
                    chunk_transcription = self.transcribe_chunk(chunk, language)
                    transcriptions.append(chunk_transcription)

                # Combine all transcriptions
                full_transcription = " ".join(transcriptions)
                return full_transcription

            else:
                # Process short audio normally
                return self.transcribe_chunk(audio, language)

        except Exception as e:
            return f"Error during transcription: {str(e)}"


# Initialize transcriber
transcriber = NigerianWhisperTranscriber()


def transcribe_audio_unified(audio_file, audio_mic, language):
    """Gradio function for transcription from either file or microphone"""
    # Determine which audio source to use
    audio_source = audio_file if audio_file is not None else audio_mic

    if audio_source is None:
        return "Please upload an audio file or record from microphone"

    try:
        result = transcriber.transcribe(audio_source, language)
        return result
    except Exception as e:
        return f"Transcription failed: {str(e)}"


def get_model_info(language):
    """Get information about the selected model"""
    model_info = {
        "Yoruba": "DereAbdulhameed/Whisper-Yoruba - Whisper model specialized for Yoruba language",
        "Hausa": "Baghdad99/saad-speech-recognition-hausa-audio-to-text - Fine-tuned Whisper model for Hausa (WER: 44.4%)",
        "Igbo": "AstralZander/igbo_ASR - Wav2Vec2-XLS-R model fine-tuned for Igbo language (WER: 51%)"
    }
    return model_info.get(language, "Model information not available")


# Create Gradio interface
with gr.Blocks(
    title="Nigerian Languages Speech Transcription",
    theme=gr.themes.Soft(),
    css="""
    .main-header {
        text-align: center;
        color: #2E7D32;
        margin-bottom: 20px;
    }
    .language-info {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    """
) as demo:

    gr.HTML("""
    <h1 class="main-header">🎤 Nigerian Languages Speech Transcription</h1>
    <p style="text-align: center; color: #666;">
        Transcribe audio in Yoruba, Hausa, and Igbo using specialized Whisper models
    </p>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # Language selection
            language_dropdown = gr.Dropdown(
                choices=["Yoruba", "Hausa", "Igbo"],
                value="Yoruba",
                label="Select Language",
                info="Choose the language of your audio file"
            )

            # Audio input options
            gr.HTML("<h3>🎵 Audio Input Options</h3>")

            with gr.Tabs():
                with gr.TabItem("📁 Upload File"):
                    audio_file = gr.Audio(
                        label="Upload Audio File",
                        type="filepath",
                        format="wav"
                    )

                with gr.TabItem("🎤 Record Speech"):
                    audio_mic = gr.Audio(
                        label="Record from Microphone",
                        type="filepath"
                    )

            # Transcribe button
            transcribe_btn = gr.Button(
                "🎯 Transcribe Audio",
                variant="primary",
                size="lg"
            )

            # Model information
            model_info_text = gr.Textbox(
                label="Model Information",
                value=get_model_info("Yoruba"),
                interactive=False,
                elem_classes="language-info"
            )

        with gr.Column(scale=2):
            # Transcription output
            transcription_output = gr.Textbox(
                label="Transcription Result",
                placeholder="Your transcription will appear here...",
                lines=10,
                max_lines=20,
                show_copy_button=True
            )

            # Usage instructions
            gr.HTML("""
            <div style="margin-top: 20px; padding: 15px; background-color: #e8f5e8; border-radius: 5px;">
                <h3>📋 How to Use:</h3>
                <ol>
                    <li>Select your target language (Yoruba, Hausa, or Igbo)</li>
                    <li><strong>Option 1:</strong> Upload an audio file (WAV, MP3, etc.)</li>
                    <li><strong>Option 2:</strong> Click the microphone tab and record speech directly</li>
                    <li>Click "Transcribe Audio" to get the text transcription</li>
                    <li>Copy the result using the copy button</li>
                </ol>
                <p><strong>Note:</strong> First-time model loading may take a few minutes.</p>
                <p><strong>Recording Tip:</strong> Speak clearly and ensure good audio quality for better transcription accuracy.</p>
                <p><strong>Long Audio:</strong> Audio longer than 25 seconds will be automatically processed in chunks.</p>
            </div>
            """)

    # Event handlers
    transcribe_btn.click(
        fn=transcribe_audio_unified,
        inputs=[audio_file, audio_mic, language_dropdown],
        outputs=transcription_output,
        show_progress=True
    )

    language_dropdown.change(
        fn=get_model_info,
        inputs=language_dropdown,
        outputs=model_info_text
    )

    # Examples section
    gr.HTML("""
    <div style="margin-top: 30px;">
        <h3>🌍 Supported Languages:</h3>
        <ul>
            <li><strong>Yoruba:</strong> Widely spoken in Nigeria, Benin, and Togo</li>
            <li><strong>Hausa:</strong> Major language in Northern Nigeria and Niger</li>
            <li><strong>Igbo:</strong> Predominantly spoken in Southeastern Nigeria</li>
        </ul>
    </div>
    """)


# Launch the application
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True,
        show_error=True
    )