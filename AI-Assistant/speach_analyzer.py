import torch
import os
import gradio as gr
import json
import openai
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

class MeetingAssistant:
    """
    A class to encapsulate the AI Meeting Assistant functionalities,
    including LLM initialization, transcription, and text generation.
    """
    def __init__(
            self,
            model="gpt-4o-mini",
            speech_model="openai/whisper-tiny.en"):
        """
        Initializes the OpenAI client and defines parameters for different LLM uses.

        :param model: The OpenAI model ID to use for text generation.
        :param speech_model: The model to use for speech-to-text transcription.
        """

        self.client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        self.OPENAI_MODEL_ID = model

        # Default parameters for main LLM (Meeting Minutes)
        self.LLM_PARAMS = {
            "temperature": 0.5,
            "top_p": 1,
            "max_tokens": 512
        }

        # Parameters for Product Assistant LLM
        self.PRODUCT_ASSISTANT_PARAMS = {
            "temperature": 0.2,
            "top_p": 0.6,
            "max_tokens": 512
        }

        # Initialize the speech-to-text pipeline here to avoid re-initializing on every call
        self.asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=speech_model,
            chunk_length_s=30,
        )

    async def call_openai_llm(self, prompt_text, params, model_id=None):
        """
        Makes a call to the OpenAI API for text generation.
        Uses the class's OPENAI_MODEL_ID by default, but can be overridden.

        :param prompt_text: The text prompt to send to the LLM.
        :param params: A dictionary of parameters for the LLM call.
        :param model_id: Optional; if provided, overrides the default model ID.
        :return: The generated text response from the LLM.
        """

        if model_id is None:
            model_id = self.OPENAI_MODEL_ID

        messages = [{"role": "user", "content": prompt_text}]

        try:
            response = await self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=params.get("temperature"),
                top_p=params.get("top_p"),
                max_tokens=params.get("max_tokens")
            )
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return "Error: Could not generate response from LLM."

    @staticmethod
    def remove_non_ascii(text):
        """
        Removes non-ASCII characters from a string.

        :param text: The input string from which non-ASCII characters will be removed.
        :return: A string containing only ASCII characters.
        """

        return ''.join(i for i in text if ord(i) < 128)

    async def product_assistant(self, ascii_transcript):
        """
        Acts as an intelligent assistant specializing in financial products.
        Processes transcripts to format financial terms and product references.

        :param ascii_transcript: The transcript text to be processed.
        :return: The adjusted transcript with financial terms formatted and a list of changes made.
        """

        system_prompt = """You are an intelligent assistant specializing in financial products;
        your task is to process transcripts of earnings calls, ensuring that all references to
        financial products and common financial terms are in the correct format. For each
        financial product or common term that is typically abbreviated as an acronym, the full term
        should be spelled out followed by the acronym in parentheses. For example, '401k' should be
        transformed to '401(k) retirement savings plan', 'HSA' should be transformed to 'Health Savings Account (HSA)' , 'ROA' should be transformed to 'Return on Assets (ROA)', 'VaR' should be transformed to 'Value at Risk (VaR)', and 'PB' should be transformed to 'Price to Book (PB) ratio'. Similarly, transform spoken numbers representing financial products into their numeric representations, followed by the full name of the product in parentheses. For instance, 'five two nine' to '529 (Education Savings Plan)' and 'four zero one k' to '401(k) (Retirement Savings Plan)'. However, be aware that some acronyms can have different meanings based on the context (e.g., 'LTV' can stand for 'Loan to Value' or 'Lifetime Value'). You will need to discern from the context which term is being referred to and apply the appropriate transformation. In cases where numerical figures or metrics are spelled out but do not represent specific financial products (like 'twenty three percent'), these should be left as is. Your role is to analyze and adjust financial product terminology in the text. Once you've done that, produce the adjusted transcript and a list of the words you've changed"""

        prompt_input = system_prompt + "\n" + ascii_transcript

        # Call OpenAI LLM for product assistant
        response_content = await self.call_openai_llm(prompt_input, self.PRODUCT_ASSISTANT_PARAMS)
        return response_content

    async def transcript_audio(self, audio_file):
        """
        Transcribes an audio file, processes the transcript with a product assistant,
        and then generates meeting minutes and tasks using an LLM.

        :params audio_file: Path to the audio file to be transcribed.
        :return: A tuple containing the generated meeting minutes and a file for download.
        """

        raw_transcript = self.asr_pipe(audio_file, batch_size=8)["text"]
        ascii_transcript = self.remove_non_ascii(raw_transcript)

        # Use the product assistant to adjust the transcript
        adjusted_transcript = await self.product_assistant(ascii_transcript)

        # Define the prompt for meeting minutes and tasks
        template = """
        Generate meeting minutes and a list of tasks based on the provided context.

        Context:
        {context}

        Meeting Minutes:
        - Key points discussed
        - Decisions made

        Task List:
        - Actionable items with assignees and deadlines
        """

        # Combine template with adjusted transcript for the final LLM call
        final_prompt_for_minutes = template.format(context=adjusted_transcript)

        # Call OpenAI LLM for meeting minutes generation
        result = await self.call_openai_llm(final_prompt_for_minutes, self.LLM_PARAMS)

        # Write the result to a file for downloading
        output_file = "meeting_minutes_and_tasks.txt"
        with open(output_file, "w") as file:
            file.write(result)

        # Return the textual result and the file for download
        return result, output_file


# Instantiate the MeetingAssistant class
assistant = MeetingAssistant()

# Gradio interface setup
audio_input = gr.Audio(sources="upload", type="filepath", label="Upload your audio file")
output_text = gr.Textbox(label="Meeting Minutes and Tasks")
download_file = gr.File(label="Download the Generated Meeting Minutes and Tasks")

# Pass the class method to Gradio
iface = gr.Interface(
    fn=assistant.transcript_audio, # Now calling the method of the instance
    inputs=audio_input,
    outputs=[output_text, download_file],
    title="AI Meeting Assistant",
    description="Upload an audio file of a meeting. This tool will transcribe the audio, fix product-related terminology, and generate meeting minutes along with a list of tasks."
)

iface.launch(server_name="0.0.0.0", server_port=5000)

