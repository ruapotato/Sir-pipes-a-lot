#!/usr/bin/env python3
"""
Application audio capture script specifically targeting audio from applications
like video players, browsers, etc. - not microphone input.
"""
import numpy as np
import argparse
import sys
import time
import signal
import subprocess
import os
import re

class AppAudioCapture:
    def __init__(self, sample_rate=16000, format="raw"):
        self.sample_rate = sample_rate
        self.format = format.lower()
        self.running = False
        self.process = None
        
        # Set up signal handling for clean termination
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
    
    def handle_signal(self, sig, frame):
        """Handle termination signals for clean shutdown"""
        print("\nCleaning up and exiting...", file=sys.stderr)
        self.stop()
        sys.exit(0)
    
    def run_command(self, cmd, shell=False):
        """Run a shell command and return output"""
        try:
            result = subprocess.run(cmd, shell=shell, text=True, capture_output=True)
            return result.stdout, result.stderr
        except Exception as e:
            print(f"Error running command: {e}", file=sys.stderr)
            return None, str(e)
    
    def find_output_source(self):
        """Find the monitor source for the default sink (output device)"""
        # Use pactl to list sinks (output devices)
        stdout, stderr = self.run_command(["pactl", "list", "sinks", "short"])
        
        if not stdout:
            print(f"Error getting sink list: {stderr}", file=sys.stderr)
            return None
            
        # Find the default sink
        default_sink = None
        stdout, stderr = self.run_command(["pactl", "get-default-sink"])
        if stdout:
            default_sink = stdout.strip()
            print(f"Default audio output: {default_sink}", file=sys.stderr)
        else:
            # If we can't get the default sink, just take the first one
            for line in stdout.splitlines():
                if line.strip():
                    default_sink = line.split()[1]
                    print(f"Using first available sink: {default_sink}", file=sys.stderr)
                    break
        
        if not default_sink:
            print("No audio output device found", file=sys.stderr)
            return None
        
        # Now find the monitor source for this sink
        monitor_source = f"{default_sink}.monitor"
        stdout, stderr = self.run_command(["pactl", "list", "sources", "short"])
        
        if not stdout:
            print(f"Error getting source list: {stderr}", file=sys.stderr)
            return None
        
        # Check if the monitor source exists
        for line in stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and monitor_source in parts[1]:
                print(f"Found monitor source: {parts[1]}", file=sys.stderr)
                return parts[1]
        
        # If we couldn't find the exact monitor, look for any monitor source
        for line in stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and ".monitor" in parts[1]:
                print(f"Found alternative monitor source: {parts[1]}", file=sys.stderr)
                return parts[1]
        
        print("No suitable monitor source found", file=sys.stderr)
        return None
    
    def start(self):
        """Start capturing application audio"""
        # Find the monitor source for application audio
        monitor_source = self.find_output_source()
        
        if not monitor_source:
            print("Failed to find a suitable audio source. Exiting.", file=sys.stderr)
            return
        
        print(f"Capturing audio from: {monitor_source}", file=sys.stderr)
        print(f"Sample rate: {self.sample_rate} Hz", file=sys.stderr)
        
        # Set up the parec command to capture audio
        cmd = [
            "parec",
            "--format=float32le",  # Use float32 format for compatibility with Whisper
            f"--rate={self.sample_rate}",
            f"--device={monitor_source}",
            "--channels=1",        # Mono
            "--latency=10",        # Low latency
            "--process-time=5"     # Process in small batches
        ]
        
        try:
            # Start parec as a subprocess
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=4096
            )
            
            self.running = True
            print("Application audio capture started", file=sys.stderr)
            print("Press Ctrl+C to stop", file=sys.stderr)
            
            # Process the output
            buffer_size = 1024  # Size to read at once
            
            if self.format == "raw":
                # For raw format, just pipe the data through
                while self.running and self.process.poll() is None:
                    data = self.process.stdout.read(buffer_size)
                    if not data:
                        break
                    
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
            
            elif self.format == "levels":
                # For levels format, process the audio to extract level information
                import struct
                
                try:
                    while self.running and self.process.poll() is None:
                        # Read a chunk of float32 data
                        data = self.process.stdout.read(buffer_size * 4)  # 4 bytes per float32
                        if not data:
                            break
                        
                        # Convert bytes to numpy array of float32
                        audio_data = np.frombuffer(data, dtype=np.float32)
                        
                        # Calculate audio levels
                        audio_level = np.max(np.abs(audio_data))
                        rms_level = np.sqrt(np.mean(np.square(audio_data)))
                        
                        # Output the levels
                        sys.stdout.write(f"{audio_level:.6f},{rms_level:.6f}\n")
                        sys.stdout.flush()
                        
                        # Occasionally print the level to stderr for monitoring
                        if hasattr(self, 'last_level_print') and time.time() - self.last_level_print > 1.0:
                            print(f"Audio level: {audio_level:.4f}", file=sys.stderr)
                            self.last_level_print = time.time()
                except ImportError:
                    print("Error: NumPy required for levels format", file=sys.stderr)
                    self.stop()
            
            # Check for any errors from parec
            if self.process.poll() is not None:
                stderr_output = self.process.stderr.read().decode()
                if stderr_output:
                    print(f"parec error: {stderr_output}", file=sys.stderr)
        
        except KeyboardInterrupt:
            print("\nCapture stopped by user", file=sys.stderr)
        except Exception as e:
            print(f"Error during capture: {e}", file=sys.stderr)
        finally:
            self.stop()
    
    def stop(self):
        """Stop the audio capture and clean up resources"""
        self.running = False
        
        if self.process is not None and self.process.poll() is None:
            try:
                # Try to terminate gracefully first
                self.process.terminate()
                time.sleep(0.5)
                
                # Force kill if still running
                if self.process.poll() is None:
                    self.process.kill()
            except Exception as e:
                print(f"Error stopping parec: {e}", file=sys.stderr)
        
        print("Application audio capture stopped", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Capture application audio output (not microphone)")
    parser.add_argument("--rate", type=int, default=16000, help="Sample rate in Hz (default: 16000)")
    parser.add_argument("--format", type=str, default="raw", 
                      choices=["raw", "levels"],
                      help="Output format (raw=binary float32, levels=max,rms)")
    
    args = parser.parse_args()
    
    # Create and start the capture
    capture = AppAudioCapture(
        sample_rate=args.rate,
        format=args.format
    )
    
    capture.start()

if __name__ == "__main__":
    main()
