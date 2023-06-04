import streamlit as st
import torch
import librosa
import sounddevice as sd
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
import requests
import json
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

import sys
import queue

# Basic page setup and definitions------------------------------------------------------------

st.set_page_config(
    page_title="Speech-to-Text Transcription App", layout="wide"
)

def _max_width_():
    max_width_str = f"max-width: 1200px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


_max_width_()

c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    st.header("")

with c32:

    st.title("")
    st.title("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")


st.text("")
c1, c2, c3 = st.columns([1, 4, 1])
with c2:
    st.markdown(
            f"""The speech to text recognition is done via the [Facebook's Wav2Vec2 model.](https://huggingface.co/facebook/wav2vec2-large-960h)"""
    )
st.text("")

# Defining Home page
def main():
    pages = {
        "Home": record,
    }

    if "page" not in st.session_state:
        st.session_state.update(
            {
                # Default page
                "page": record,
            }
        )

    pages["Home"]()

# recording -------------------------------------------------

# define queue 
q = queue.Queue()

# callback function
def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

# recording voice from microphone 
def record():
    c1, c2, c3 = st.columns([1, 4, 1])
    with c2:
        transcript=""
        start_button=st.button("Start Recording")

        if start_button:
            stop_button=st.button("Stop Recording")
            with sf.SoundFile("temp_input_storage.wav", mode='w', samplerate=16000, channels=1) as file:
                with sd.InputStream(samplerate=16000, channels=1, callback=callback):
                    while True:
                        file.write(q.get())
                        file_size = sys.getsizeof(file)
                        if file_size < (2 * 1024 * 1024): # File less than 2MB
                            transcript=transcripter()
                            st.write(transcript)
                        else:
                            download_transcript=st.download_button(
                                "Download the transcription",
                                transcript,
                                file_name=None,
                                mime=None,
                                key=None,
                                help=None,
                                on_click=None,
                                args=None,
                                kwargs=None,
                            )
                            break
                        
                        # stop the recording on button click
                        if stop_button:
                            download_transcript=st.download_button(
                                "Download the transcription",
                                transcript,
                                file_name=None,
                                mime=None,
                                key=None,
                                help=None,
                                on_click=None,
                                args=None,
                                kwargs=None,
                            )
                            break

def transcripter():
    # Load your API key from an environment variable or secret management service
    speech, rate = librosa.load("temp_input_storage.wav", sr=16000)
    # api_token = "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"
    # headers = {"Authorization": f"Bearer {api_token}"}
    # API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-base-960h"
    # def query(data):
    #     response = requests.request("POST", API_URL, headers=headers, data=data)
    #     return json.loads(response.content.decode("utf-8"))
    
    # data = query(speech.all())

    # values_view = data.values()
    # value_iterator = iter(values_view)
    # text_value = next(value_iterator)
    # text_value = text_value.lower()

    # return text_value
    input_values = tokenizer(speech, return_tensors='pt').input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])
    return transcription
        
if __name__ == "__main__":
    main()