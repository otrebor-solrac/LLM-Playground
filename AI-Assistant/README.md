# AI Meeting Assistant

## Introduction

The AI Meeting Assistant is a Python-based application designed to streamline the process of documenting meetings. It leverages advanced Artificial Intelligence models to automatically transcribe audio recordings, intelligently process and correct financial terminology within the transcript, and then generate concise meeting minutes along with actionable task lists. This tool is built to enhance productivity by automating tedious administrative tasks, allowing users to focus more on the content of their discussions.

## Features

* **Audio Transcription:** Converts spoken words from audio files into written text using a powerful Automatic Speech Recognition (ASR) model (Whisper).

* **Financial Terminology Correction:** Utilizes an OpenAI LLM (gpt-4o-mini) to identify and correctly format common financial terms and product acronyms (e.g., "401k" to "401(k) retirement savings plan", "HSA" to "Health Savings Account (HSA)"). It intelligently discerns context for ambiguous acronyms like "LTV".

* **Meeting Minutes Generation:** Generates structured meeting minutes, including key discussion points and decisions made, based on the processed transcript.

* **Task List Creation:** Automatically extracts actionable items, often with suggested assignees and deadlines, from the meeting transcript.

* **Downloadable Output:** Provides the generated meeting minutes and task list as a downloadable text file.

* **User-Friendly Interface:** Built with Gradio for an intuitive web-based user experience.

* **Modular Design:** Implemented as a Python class for better code organization, reusability, and maintainability.

## How It Works

The application follows a sequential pipeline:

1.  **Audio Upload:** Users upload an audio file of their meeting via the Gradio interface.

2.  **Speech-to-Text Transcription:** The uploaded audio is processed by the Hugging Face `transformers` pipeline using the `openai/whisper-tiny.en` model to generate a raw text transcript.

3.  **Transcript Cleaning:** Non-ASCII characters are removed from the raw transcript.

4.  **Financial Terminology Adjustment:** The cleaned transcript is then fed to an OpenAI `gpt-4o-mini` model (`product_assistant`) which applies predefined rules to correct and expand financial terms and acronyms based on context.

5.  **Meeting Minutes & Task Generation:** The adjusted transcript is passed to another OpenAI `gpt-4o-mini` instance, which generates the comprehensive meeting minutes and a structured task list.

6.  **Output Display & Download:** The generated text is displayed in the Gradio interface, and a downloadable text file is provided.

## Setup

### Prerequisites

* Python 3.8 or higher

* An OpenAI API Key.
    * Set this as an environment variable named `OPENAI_API_KEY` (e.g., in a `.env` file).

### Installation

1.  **Clone the repository (or save the code locally):**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate 
    ```

3.  **Install the required Python packages:**

    ```bash
    pip install torch transformers openai gradio python-dotenv
    ```

4.  **Create a `.env` file:**
    In the root directory of your project, create a file named `.env` and add your OpenAI API key:

    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

    Replace `"your_openai_api_key_here"` with your actual OpenAI API key.

## Usage

To run the AI Meeting Assistant:

1.  **Ensure your virtual environment is active.**

2.  **Navigate to the directory containing the `main.py` (or whatever you named your script) file.**

3.  **Execute the script:**

    ```bash
    python your_script_name.py
    ```

4.  **Access the Gradio interface:**
    The script will output a local URL (e.g., `http://0.0.0.0:5000`) where you can access the web interface in your browser.

5.  **Upload an audio file (.wav format recommended for best performance), and the assistant will process it.**

## Key Technologies

* **Python:** The core programming language.

* **Hugging Face `transformers`:** For the Automatic Speech Recognition (ASR) pipeline (Whisper model).

* **OpenAI API:** For powerful Large Language Model (LLM) capabilities (gpt-4o-mini) for text generation and intelligent formatting.

* **Gradio:** For creating the interactive web user interface.

* **`python-dotenv`:** For secure management of environment variables.
