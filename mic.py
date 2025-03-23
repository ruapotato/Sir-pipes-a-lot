#!/usr/bin/env python3
"""
Microphone audio capture script that streams audio data to stdout
for processing by Whisper ASR and speech detection systems.
"""
import sounddevice as sd
import numpy as np
import argparse
import sys
import time
import signal

class MicrophoneCapture:
    def __init__(self, sample_rate=16000, channels=1, block_size=1024, format="raw"):
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.format = format.lower()
        self.running = False
        self.stream = None
        
        # Set up signal handling for clean termination
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
    
    def handle_signal(self, sig, frame):
        """Handle termination signals"""
        self.stop()
        sys.exit(0)
    
    def audio_callback(self, indata, frames, time_info, status):
        """Process incoming audio data"""
        if status:
            print(f"Status: {status}", file=sys.stderr)
        
        try:
            # Convert to mono if necessary
            if self.channels == 1 and len(indata.shape) > 1 and indata.shape[1] > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata.flatten()
            
            # Normalize audio to [-1, 1] range
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 1:
                audio_data = audio_data / 32768.0
            
            # Process based on format
            if self.format == "raw":
                # Output raw binary float32 data for Whisper processing
                sys.stdout.buffer.write(audio_data.tobytes())
                sys.stdout.buffer.flush()
            
            elif self.format == "levels":
                # Calculate and output audio levels for speech detection
                audio_level = np.max(np.abs(audio_data))
                rms_level = np.sqrt(np.mean(np.square(audio_data)))
                sys.stdout.write(f"{audio_level:.6f},{rms_level:.6f}\n")
                sys.stdout.flush()
                
        except Exception as e:
            print(f"Error in audio callback: {e}", file=sys.stderr)
    
    def start(self):
        """Start the audio capture"""
        try:
            # List available devices
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            print(f"Using input device: {devices[default_input]['name']}", file=sys.stderr)
            
            # Create and start stream
            self.stream = sd.InputStream(
                device=default_input,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                callback=self.audio_callback
            )
            
            self.running = True
            self.stream.start()
            
            # Keep the script running
            while self.running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            self.stop()
    
    def stop(self):
        """Stop the audio capture"""
        self.running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

def main():
    parser = argparse.ArgumentParser(description="Capture microphone audio and output to stdout for ASR processing")
    parser.add_argument("--rate", type=int, default=16000, help="Sample rate in Hz (default: 16000)")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels (default: 1)")
    parser.add_argument("--blocksize", type=int, default=1024, help="Block size for audio capture (default: 1024)")
    parser.add_argument("--format", type=str, default="raw", 
                      choices=["raw", "levels"],
                      help="Output format (raw=binary float32 for Whisper, levels=max,rms for speech detection)")
    
    args = parser.parse_args()
    
    # Create and start the capture
    capture = MicrophoneCapture(
        sample_rate=args.rate,
        channels=args.channels,
        block_size=args.blocksize,
        format=args.format
    )
    
    capture.start()

if __name__ == "__main__":
    main()
