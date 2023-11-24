import whisper
from pytube import YouTube
import gradio as gr
import os
import re
import logging
from easygoogletranslate import EasyGoogleTranslate
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize 
from typing import Tuple, Optional, List
 
class VideoTranscriber: 

    def __init__(self) -> None:

        """
        Initializes the VideoTranscriber class by loading the whisper model and downloading NLTK resources.
        """      
        self.model = whisper.load_model("base")
        nltk.download("punkt")
        nltk.download("stopwords")

    def text_summarizer(text: str) -> str:

        """
        Summarizes the given text using a simple algorithm based on word frequency.

        Args:
            text (str): The input text to be summarized.

        Returns:
            str: The summary of the input text.
        """      
        # Tokenize the text into sentences and words
        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        # Remove stopwords (common words that don't add much meaning)
        stop_words = set(stopwords.words("english"))
        words = [word for word in words if word.lower() not in stop_words]

        # Calculate word frequency
        word_frequency = {}
        for word in words:
            if word not in word_frequency:
                word_frequency[word] = 1
            else:
                word_frequency[word] += 1

        # Calculate sentence scores based on word frequency
        sentence_scores = {}
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word in word_frequency:
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = word_frequency[word]
                    else:
                        sentence_scores[sentence] += word_frequency[word]

        # Get the top 'num_sentences' sentences with highest scores
        summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:4]

        # Combine the selected sentences to form the summary
        summary = " ".join(summary_sentences)

        return summary


    def get_text(self, url: str, language: str) -> Tuple[str, str]:

        """
        Transcribes the audio of a YouTube video and translates the result to the specified language.

        Args:
            url (str): The URL of the YouTube video.
            language (str): The target language for translation.

        Returns:
            Tuple[str, str]: A tuple containing the transcribed text and its summary.
        """      
        #try:
        if url != '':
            output_text_transcribe = ''

        yt = YouTube(url)
        #video_length = yt.length --- doesn't work anymore - using byte file size of the audio file instead now
        #if video_length < 5400:
        video = yt.streams.filter(only_audio=True).first()
        out_file=video.download(output_path=".")

        file_stats = os.stat(out_file)
        logging.info(f'Size of audio file in Bytes: {file_stats.st_size}')

        if file_stats.st_size <= 30000000:
            base, ext = os.path.splitext(out_file)
            new_file = base+'.mp3'
            os.rename(out_file, new_file)
            a = new_file
            result = self.model.transcribe(a)
            text =  result['text'].strip()
            translator = EasyGoogleTranslate(
                source_language='en',
                target_language=language,
                timeout=10
            )
            result = translator.translate(text)
            summary = self.text_summarizer(result)
            return result, summary

    def gradio_interface(self):

        """
        Sets up and launches the Gradio interface for YouTube video transcription.
        """      
        with gr.Blocks(css="style.css",theme= 'freddyaboulton/test-blue') as demo:
            with gr.Column(elem_id="col-container"):
                gr.HTML("""<center><h1 style="color:#fff">YouTube Video Transcriber </h1></center>""")
                with gr.Row():
                    with gr.Column(scale=0.4):
                        language = gr.Dropdown(
                        ["en","ta","te","hi","ml"], label="Select Language"
                    )
                    with gr.Column(scale=0.6):
                        input_text_url = gr.Textbox(placeholder='Youtube video URL', label='YouTube URL',elem_classes="textbox")
                with gr.Row():
                    result_button_transcribe = gr.Button('Transcribe')

                with gr.Row():
                    output_text_transcribe = gr.Textbox(placeholder='Transcript of the YouTube video.', label='Transcript',lines=10)

                with gr.Row():
                    output_text_summary = gr.Textbox(placeholder='Summary of the YouTube video transcript.', label='Summary',lines=5)

            result_button_transcribe.click(self.get_text, inputs = [input_text_url,language], outputs = [output_text_transcribe,output_text_summary] )

            demo.launch(share = True)

if __name__ == "__main__":
    transcriber = VideoTranscriber()
    transcriber.gradio_interface()            
