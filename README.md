# Sir-pipes-a-lot

A real-time voice control and analysis framework utilizing AI for transcription and processing.

## Overview

Sir-pipes-a-lot is an advanced framework designed to control and analyze voice data in real-time. It captures both system audio output and microphone input, providing real-time transcription, voice activity detection, and AI-powered analysis. This makes it ideal for applications like call center monitoring, meeting transcription, and voice-controlled systems.

## Features

- Real-time audio capture from both system output and microphone input
- Voice Activity Detection (VAD) using WebRTC
- Real-time transcription using Whisper AI
- CUDA-accelerated processing for improved performance
- GTK4-based user interface
- Support for virtual audio device routing
- Extensible architecture for custom processing pipelines

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (recommended)
- Audio input/output devices

### System Dependencies
```bash
sudo apt-get install -y \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gtk-4.0 \
    libgirepository1.0-dev \
    gcc \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    gir1.2-gtk-3.0 \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc
```

### Python Environment
Python 3.x with virtual environment support. The project requires system GTK bindings, so create your virtual environment with:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Sir-pipes-a-lot.git
cd Sir-pipes-a-lot
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

3. Activate the virtual environment:
```bash
source venv/bin/activate
```

## Bits
system_audio.py dumps audio going to the speakers to STDOUT. Example: Videos, applications
mic.py dumps audio coming from the microphone to STDOUT.
#TODO: segment.py Uses WebRTC VAD on STDIN from mic or system_audio, and buffers the output and formats for whisper.py
#TODO: whisper.py steams data over STDIN from segment.py, processing the input when a pause in the input is detected

## Usage

Run the application:
```bash
# To dump system audio as text
python3 ./system_audio.py --format raw | python3 ./segment.py | python3 ./whisper.py
# To dump mic audio as text
python3 ./mic.py --format raw | python3 ./segment.py | python3 ./whisper.py
# To track both:
#TODO: python main.py
```

The GUI will provide controls for:
- Starting/stopping audio capture
- Monitoring transcription status
- Viewing real-time analysis

## Architecture

The framework consists of several key components:

- AudioProcessor: Handles real-time audio capture and processing
- VoiceActivityDetector: Identifies speech segments in audio streams
- AudioTranscriber: Manages Whisper-based transcription
- MainWindow: GTK4-based user interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Author

David Hamner

## Acknowledgments

- OpenAI Whisper for transcription
- WebRTC VAD for voice activity detection
- GTK team for the GUI framework
- NVIDIA for CUDA support

## Support

For issues, questions, or contributions, please file an issue in the GitHub repository.

---
*Sir-pipes-a-lot: Because every voice deserves to be heard, analyzed, and understood.*
