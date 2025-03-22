"""Whisper-based speech recognition processor."""
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import logging
import warnings
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

class WhisperProcessor:
    def __init__(self, model_name="openai/whisper-tiny.en"):
        """Initialize the Whisper processor."""
        print("Initializing Whisper processor...", flush=True)
        self.model_loaded = False
        self.model_name = model_name
        self.setup_model()
        
    def setup_model(self):
        """Initialize the Whisper model and processor."""
        try:
            # Setup device
            print("DEBUG: Checking CUDA availability...", flush=True)
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"DEBUG: Using device: {self.device}", flush=True)

            # Load model - Match the types!
            print(f"DEBUG: Loading model: {self.model_name}", flush=True)
            model_dtype = torch.float32  # Use float32 consistently
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                torch_dtype=model_dtype
            )
            self.model.to(self.device)
            print("DEBUG: Model loaded", flush=True)

            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            print("DEBUG: Processor loaded", flush=True)

            # Test transcription
            dummy_audio = np.zeros(4000, dtype=np.float32)
            test_result = self.transcribe(dummy_audio)
            print(f"DEBUG: Test transcription result: {test_result}", flush=True)
            
            self.model_loaded = True
            print("DEBUG: ✅ Whisper model initialized successfully", flush=True)
            logger.info("✅ Whisper model initialized successfully")

        except Exception as e:
            print(f"DEBUG: ❌ Error initializing Whisper: {e}", flush=True)
            import traceback
            print(f"DEBUG: Stack trace: {traceback.format_exc()}", flush=True)
            self.model_loaded = False

    def transcribe(self, audio_data):
        """Process audio data and return transcribed text."""
        if not self.model_loaded:
            print("DEBUG: Model not loaded", flush=True)
            return "Model not loaded"
            
        try:
            # Basic checks
            if audio_data is None or len(audio_data) == 0:
                print("DEBUG: Empty audio data", flush=True)
                return None

            # Check audio level
            audio_level = np.max(np.abs(audio_data))
            if audio_level < 0.01:
                print("DEBUG: Audio too quiet, skipping", flush=True)
                return None
                
            # Format audio
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            audio_data = np.clip(audio_data / np.max(np.abs(audio_data)) * 0.95, -1, 1)
            
            # Process audio using processor
            input_features = self.processor(
                audio_data, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features
            
            # Ensure input features are float32 to match model
            input_features = input_features.to(dtype=torch.float32, device=self.device)
            
            # Generate transcription
            with torch.inference_mode():
                predicted_ids = self.model.generate(input_features)
            
            # Decode
            transcribed_text = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            return transcribed_text
            
        except Exception as e:
            print(f"DEBUG: Error in transcription: {e}", flush=True)
            import traceback
            print(f"DEBUG: Trace: {traceback.format_exc()}", flush=True)
            return f"Error: {str(e)}"
