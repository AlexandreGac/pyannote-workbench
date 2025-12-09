# Pyannote Workbench: Human-in-the-Loop Diarization

Pyannote Workbench is an interactive web application designed to explore **Human-in-the-Loop (HITL) Speaker Diarization**.

This application allows users to upload audio files, process them using the Pyannote.ai API, visualize the speaker embeddings in 2D space (using t-SNE), and interactively correct speaker assignation errors using a Kanban-style interface.

---

## Application Features

*   **Automated Diarization**: Integration with Pyannote.ai's precision models via API.
*   **Interactive Kanban Board**: Drag-and-drop interface to correct speaker labels, merge segments, and split merged audio.
*   **Embedding Visualization**: Real-time 2D visualization of speaker voiceprints using t-SNE (t-Distributed Stochastic Neighbor Embedding).
*   **Assisted Clustering**: Apply Spectral Clustering on selected segments to automatically separate mixed speakers based on embedding affinity.
*   **Audio Waveform**: Interactive waveform with region highlighting.
*   **Export**: Download the corrected diarization annotations as JSON.

---

## Prerequisites

Before installing the Python dependencies, you must ensure the following system-level tools are installed.

### 1. FFmpeg
This application uses `pydub` for audio processing, which requires **FFmpeg** to be installed and available in your system's PATH.

*   **MacOS (via Homebrew):**
    ```bash
    brew install ffmpeg
    ```
*   **Ubuntu/Debian Linux:**
    ```bash
    sudo apt update && sudo apt install ffmpeg
    ```
*   **Windows:**
    1.  Download the build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
    2.  Extract the ZIP file.
    3.  Add the `bin` folder (e.g., `C:\ffmpeg\bin`) to your System Environment Variables -> Path.
    4.  Restart your terminal/PowerShell.

### 2. Pyannote API Key
You need a valid API key to access the diarization models.

---

## Installation

Follow these steps to set up the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/pyannote-workbench.git
cd pyannote-workbench
```

### 2. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

**MacOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
Run the installation command:
```bash
pip install -r requirements.txt
```

---

## How to Run

1.  **Start the Application:**
    Ensure your virtual environment is active, then run:
    ```bash
    python app.py
    ```

2.  **Access the Interface:**
    The application utilizes a threading timer to automatically open your default web browser to:
    `http://127.0.0.1:5000`

3.  **Workflow:**
    *   Paste your **Pyannote API Token** in the top-left input field.
    *   Select an **Audio File** (WAV, MP3, etc.).
    *   Click the **Play Button** (Start Process).
    *   Wait for the upload and diarization jobs to complete.
    *   Use the Kanban board to correct speaker errors or the "Embedding Space" panel to analyze clusters.

---

## AI Disclaimer

Please note that portions of the code and documentation for this project were generated with the assistance of Artificial Intelligence tools. This project serves as an experimental application to demonstrate HITL workflows.
