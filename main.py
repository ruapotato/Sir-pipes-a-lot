#!/usr/bin/env python3
import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, GLib, Gio
import sounddevice as sd
import numpy as np
import webrtcvad
import wave
import threading
import queue
import logging
import asyncio
import sys
import time  # Add missing import
from dataclasses import dataclass
from typing import Optional, List, Tuple
from whisper_processor import WhisperProcessor  # Import the new WhisperProcessor

# Configure logging - FIXED to prevent duplicate logs
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[])  # No handlers here, we'll add them manually

# Create file handler that logs to a file
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Get the root logger and add handlers
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Get logger for this module
logger = logging.getLogger(__name__)
logger.info("Application starting...")

@dataclass
class AudioSegment:
    """Represents a segment of audio with its metadata"""
    audio_data: np.ndarray
    timestamp: float
    is_speech: bool
    source: str  # 'mic' or 'system'

class VoiceActivityDetector:
    def __init__(self, sample_rate: int = 16000):
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3
        self.sample_rate = sample_rate
        self.frame_duration = 30  # ms
        self.frame_size = int(sample_rate * self.frame_duration / 1000)
        logger.info(f"VAD initialized with sample rate {sample_rate} Hz, frame size {self.frame_size}")

    def is_speech(self, audio_frame: np.ndarray) -> bool:
        """Detect if the audio frame contains speech"""
        try:
            # Check if audio level is high enough
            audio_level = np.max(np.abs(audio_frame))
            if audio_level < 0.01:  # Very quiet
                return False
                
            # Ensure audio is the right format for WebRTC VAD
            # Convert to 16-bit PCM
            audio_frame = (audio_frame * 32768).astype(np.int16)
            
            # Ensure we have the right number of samples
            samples_per_window = int(self.sample_rate * 0.03)  # 30ms
            if len(audio_frame) >= samples_per_window:
                audio_frame = audio_frame[:samples_per_window]
            else:
                # Pad with zeros if needed
                audio_frame = np.pad(audio_frame, (0, samples_per_window - len(audio_frame)))
            
            result = self.vad.is_speech(audio_frame.tobytes(), self.sample_rate)
            return result
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return False

class AudioProcessor:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        logger.info(f"Initializing AudioProcessor with sample rate {sample_rate} Hz")
        
        self.vad = VoiceActivityDetector(sample_rate=sample_rate)
        logger.info("VAD initialized")
        
        self.whisper = None
        self.window = None
        
        # Configure audio settings
        self.block_size = int(sample_rate * 0.03)  # 30ms blocks
        self.channels = 1  # Mono audio
        
        # Setup transcription machinery
        self.audio_buffer = []
        self.buffer_timeout = None
        self.is_recording = False
        self.transcription_queue = queue.Queue()
        self.segment_counter = 0
        
        # Start the transcription thread
        self.transcription_thread = threading.Thread(target=self._process_transcription)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
        logger.info("Transcription thread started")
        
        # Initialize Whisper in a separate thread
        threading.Thread(target=self._init_whisper, daemon=True).start()

    def process_audio(self, indata: np.ndarray, frames: int, 
                     time: float, status: sd.CallbackFlags, source: str):
        """Process incoming audio data"""
        if status:
            logger.warning(f"Audio callback status: {status}")
            return

        try:
            # Convert audio to mono if necessary and normalize
            if len(indata.shape) > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata.flatten()
            
            # Normalize audio to [-1, 1] range if not already
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 1:
                audio_data = audio_data / 32768.0
            
            # Debug audio levels occasionally
            max_level = np.max(np.abs(audio_data))
            if max_level > 0.01:  # Only log if there's significant audio
                logger.debug(f"Audio level ({source}): {max_level:.4f}")
            
            # Process in chunks suitable for VAD
            chunk_size = int(self.sample_rate * 0.03)  # 30ms chunks
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    break
                
                # Check for speech
                is_speech = self.vad.is_speech(chunk)
                
                if is_speech:
                    if len(self.audio_buffer) == 0:
                        logger.debug(f"Speech detected, starting to buffer ({source})")
                    self.audio_buffer.append(chunk)
                    
                    # Reset the timeout timer when speech is detected
                    if self.buffer_timeout is not None:
                        self.buffer_timeout = None
                        
                elif len(self.audio_buffer) > 0:
                    # Start timeout for processing the buffer
                    if self.buffer_timeout is None:
                        self.buffer_timeout = 0
                    else:
                        self.buffer_timeout += 1
                        
                    # If we've had a significant pause, process the buffer
                    if self.buffer_timeout >= 10:  # ~300ms of silence
                        
                        # Process accumulated buffer if it's large enough
                        if len(self.audio_buffer) >= 5:  # At least ~150ms of speech
                            logger.debug(f"Processing speech segment of {len(self.audio_buffer)} chunks ({source})")
                            if len(self.audio_buffer) >= 5 and all(chunk is not None and len(chunk) > 0 for chunk in self.audio_buffer):
                                complete_segment = np.concatenate(self.audio_buffer)
                                segment = AudioSegment(
                                    audio_data=complete_segment,
                                    timestamp=time,
                                    is_speech=True,
                                    source=source
                                )
                                self.transcription_queue.put(segment)
                        else:
                            logger.debug(f"Discarding short segment of {len(self.audio_buffer)} chunks ({source})")
                            
                        # Reset the buffer and timeout
                        self.audio_buffer = []
                        self.buffer_timeout = None
                    
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return

    def _init_whisper(self):
        """Initialize Whisper in a separate thread"""
        try:
            logger.info("Starting Whisper initialization...")
            self.whisper = WhisperProcessor()
            logger.info("Whisper model successfully initialized")
            
            if hasattr(self, 'window') and self.window is not None:
                GLib.idle_add(self.window.status_label.set_text, "Status: Ready (Whisper loaded)")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {e}")
            if hasattr(self, 'window') and self.window is not None:
                GLib.idle_add(self.window.status_label.set_text, f"Error: Whisper init failed: {str(e)}")

    def _process_transcription(self):
        """Background thread for processing transcriptions"""
        logger.info("Transcription processing thread started")
        while True:
            try:
                audio_segment = self.transcription_queue.get()
                
                # Skip if we got None
                if audio_segment is None:
                    self.transcription_queue.task_done()
                    continue

                # Check if whisper is initialized
                if self.whisper is None or not getattr(self.whisper, 'model_loaded', False):
                    logger.warning("Whisper not yet initialized, skipping transcription")
                    if hasattr(self, 'window') and self.window is not None:
                        GLib.idle_add(self.window.status_label.set_text, "Status: Waiting for Whisper to initialize...")
                    self.transcription_queue.task_done()
                    continue
                
                self.segment_counter += 1
                segment_id = self.segment_counter
                logger.info(f"Processing segment #{segment_id} from {audio_segment.source}, length: {len(audio_segment.audio_data)}")
                
                print(f"DEBUG: Starting transcription for segment #{segment_id}", flush=True)
                
                # Direct synchronous call to transcribe
                text = self.whisper.transcribe(audio_segment.audio_data)
                
                if text:
                    logger.info(f"Transcription #{segment_id} ({audio_segment.source}): {text}")
                    if hasattr(self, 'window') and self.window is not None:
                        GLib.idle_add(self.update_transcription_ui, text, audio_segment.source)
                        GLib.idle_add(self.window.status_label.set_text, "Status: Recording")
                else:
                    logger.warning(f"Empty transcription result for segment #{segment_id}")

            except Exception as e:
                logger.error(f"Error in transcription: {e}")
                if hasattr(self, 'window') and self.window is not None:
                    GLib.idle_add(self.window.status_label.set_text, f"Status: Error: {str(e)}")
            finally:
                try:
                    if audio_segment is not None:
                        self.transcription_queue.task_done()
                except Exception:
                    pass  # Ignore errors from task_done

    def update_transcription_ui(self, text: str, source: str):
        """Update UI with transcription results"""
        if hasattr(self, 'window') and self.window is not None:
            if source == 'mic':
                self.window.update_mic_transcription(text)
            else:
                self.window.update_system_transcription(text)
        return False  # Required for GLib.idle_add


class MainWindow(Gtk.ApplicationWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Set up the window
        self.set_title("Sir-pipes-a-lot")
        self.set_default_size(800, 600)
        logger.info("Creating main window")

        # Create main layout
        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.set_child(self.box)

        # Add controls and transcription areas
        self.setup_controls()
        self.setup_transcription_areas()
        
        # Initialize audio processing
        self.audio_processor = AudioProcessor()
        self.audio_processor.window = self  # Set window reference AFTER creating the processor
        
        # Initialize streams after a short delay to allow UI to appear
        GLib.timeout_add(500, self.delayed_stream_setup)
        
        logger.info("Main window initialized")

    def delayed_stream_setup(self):
        """Set up audio streams after a short delay"""
        self.setup_audio_streams()
        return False  # Don't repeat

    def setup_controls(self):
        """Set up the GUI controls"""
        # Control buttons
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.box.append(button_box)

        # Start/Stop button
        self.toggle_button = Gtk.Button(label="Start Recording")
        self.toggle_button.connect("clicked", self.on_toggle_recording)
        button_box.append(self.toggle_button)

        # Status label
        self.status_label = Gtk.Label(label="Status: Initializing...")
        button_box.append(self.status_label)
        
        logger.info("Controls set up")

    def setup_transcription_areas(self):
        """Set up areas for displaying transcriptions"""
        # Microphone transcription
        mic_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.box.append(mic_box)
        
        mic_label = Gtk.Label(label="Microphone Input:")
        mic_box.append(mic_label)
        
        self.mic_text = Gtk.TextView()
        self.mic_text.set_editable(False)
        self.mic_text.set_wrap_mode(Gtk.WrapMode.WORD)
        mic_scroll = Gtk.ScrolledWindow()
        mic_scroll.set_vexpand(True)
        mic_scroll.set_child(self.mic_text)
        mic_box.append(mic_scroll)

        # System output transcription
        sys_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.box.append(sys_box)
        
        sys_label = Gtk.Label(label="System Output:")
        sys_box.append(sys_label)
        
        self.sys_text = Gtk.TextView()
        self.sys_text.set_editable(False)
        self.sys_text.set_wrap_mode(Gtk.WrapMode.WORD)
        sys_scroll = Gtk.ScrolledWindow()
        sys_scroll.set_vexpand(True)
        sys_scroll.set_child(self.sys_text)
        sys_box.append(sys_scroll)
        
        logger.info("Transcription areas set up")

    def setup_audio_streams(self):
        """Set up audio input and output streams"""
        try:
            # List available devices for debugging
            logger.info("Available audio devices:")
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                logger.info(f"Device {i}: {dev['name']} (in={dev['max_input_channels']}, out={dev['max_output_channels']})")
            
            # Get default devices
            default_input = sd.default.device[0]
            default_output = sd.default.device[1]
            logger.info(f"Default input device: {default_input}")
            logger.info(f"Default output device: {default_output}")
            
            # Microphone input stream
            self.mic_stream = sd.InputStream(
                device=default_input,
                channels=1,
                samplerate=16000,
                blocksize=8000,
                callback=lambda *args: self.audio_processor.process_audio(*args, source='mic')
            )
            logger.info(f"Microphone stream configured with device {default_input}")

            # Find the best device for system audio
            system_device = None
            loopback_keywords = ['loop', 'monitor', 'mix', 'output', 'playback']
            
            # First try to find a dedicated loopback device
            for i, dev in enumerate(devices):
                dev_name = dev['name'].lower()
                if any(keyword in dev_name for keyword in loopback_keywords) and dev['max_input_channels'] > 0:
                    system_device = i
                    logger.info(f"Found loopback device: {i}: {dev['name']}")
                    break
            
            # If no loopback device found, try the system default output device
            if system_device is None:
                # Some systems allow recording from an output device
                system_device = default_output if devices[default_output]['max_input_channels'] > 0 else default_input
                logger.warning(f"No loopback device found. Using device {system_device} for system audio")
            
            # Setup system stream
            try:
                self.system_stream = sd.InputStream(
                    device=system_device,
                    channels=1,
                    samplerate=16000,
                    blocksize=8000,
                    callback=lambda *args: self.audio_processor.process_audio(*args, source='system')
                )
                logger.info(f"System audio stream configured with device {system_device}")
            except Exception as e:
                logger.error(f"Failed to set up system audio stream: {e}")
                self.system_stream = None
                logger.warning("Continuing without system audio capture")
                
            # Update status
            self.status_label.set_text("Status: Ready - Waiting for Whisper to initialize...")
                
        except Exception as e:
            logger.error(f"Error setting up audio streams: {e}")
            self.status_label.set_text(f"Error: {str(e)}")

    def update_mic_transcription(self, text: str):
        """Update microphone transcription text"""
        buffer = self.mic_text.get_buffer()
        end_iter = buffer.get_end_iter()
        buffer.insert(end_iter, text + "\n")
        logger.debug(f"Updated mic transcription with: {text}")
        
        # Auto-scroll to bottom
        self.mic_text.scroll_to_iter(buffer.get_end_iter(), 0.0, True, 0.0, 1.0)

    def update_system_transcription(self, text: str):
        """Update system output transcription text"""
        buffer = self.sys_text.get_buffer()
        end_iter = buffer.get_end_iter()
        buffer.insert(end_iter, text + "\n")
        logger.debug(f"Updated system transcription with: {text}")
        
        # Auto-scroll to bottom
        self.sys_text.scroll_to_iter(buffer.get_end_iter(), 0.0, True, 0.0, 1.0)

    def on_toggle_recording(self, button):
        """Toggle recording state"""
        if not self.audio_processor.is_recording:
            try:
                logger.info("Starting recording...")
                self.mic_stream.start()
                logger.info("Microphone stream started")
                
                if hasattr(self, 'system_stream') and self.system_stream is not None:
                    self.system_stream.start()
                    logger.info("System stream started")
                
                self.audio_processor.is_recording = True
                button.set_label("Stop Recording")
                self.status_label.set_text("Status: Recording")
                logger.info("Recording started successfully")
                
                # Add a test message in the UI to confirm it's working
                self.update_mic_transcription("Recording started. Speak to see transcription.")
                
            except Exception as e:
                logger.error(f"Error starting recording: {e}")
                self.status_label.set_text(f"Error: {str(e)}")
        else:
            try:
                logger.info("Stopping recording...")
                self.mic_stream.stop()
                logger.info("Microphone stream stopped")
                
                if hasattr(self, 'system_stream') and self.system_stream is not None:
                    self.system_stream.stop()
                    logger.info("System stream stopped")
                    
                self.audio_processor.is_recording = False
                button.set_label("Start Recording")
                self.status_label.set_text("Status: Stopped")
                logger.info("Recording stopped successfully")
                
            except Exception as e:
                logger.error(f"Error stopping recording: {e}")

class VoiceControlApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="com.example.voicecontrol")
        logger.info("Voice Control App initialized")

    def do_activate(self):
        win = MainWindow(application=self)
        win.present()
        logger.info("Main window presented")

def main():
    logger.info("Starting application...")
    app = VoiceControlApp()
    return app.run(None)

if __name__ == "__main__":
    main()
