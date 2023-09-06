#Import the libraries
import streamlit as st
import pytesseract
import cv2
from PIL import Image
import openai
import constant
import numpy as np
from langdetect import detect
from gtts import gTTS



# Set OpenAI API key
openai.api_key = constant.api_key


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
config = "-l eng+jpn+kor+rus+chi_sim+vie+thai+hin+mal"


# Streamlit UI
st.title("Language Translation App")

# Sidebar for selecting translation options
st.sidebar.write("Upload an image for translation:")
source_language = st.sidebar.selectbox("Enter Source Language:",["English","French","German","Malayalam","Chinese","Japanese","Italian","Hindi"])
uploaded_image = st.sidebar.file_uploader("Upload Image", type=["jpg", "png"])
target_language = st.sidebar.selectbox("Enter Target Language:",["English","French","German","Hindi","Malayalam"])

if source_language == "English":
    lan = "en"
elif source_language == "French":
    lan = "fr"
elif source_language == "German":
    lan = 'de'
elif source_language == "Malayalam":
    lan = 'ml'
elif source_language == "Hindi":
    lan = 'hi'
elif source_language == 'Italian':
    lan = 'it' 
elif source_language == "Chinese":
    lan = 'zh-cn'
elif source_language == "Japanese":
    lan = 'ja'

if uploaded_image and target_language:

    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert PIL Image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Perform OCR using Tesseract
    extracted_text = pytesseract.image_to_string(opencv_image, config = config)
    st.write("Extracted Text:")
    st.write(extracted_text)
        
    
    # Convert extracted text to speech using gTTS

    def detect_language(text):
        return detect(text)
        
    #lan = detect_language(extracted_text)

    tts = gTTS(text=extracted_text, lang=lan)
    tts.save("original_speech.mp3")
    st.audio("original_speech.mp3")

    ok = st.button("TRANSLATE")
    if ok:
    
        # Perform translation using OpenAI
        if target_language != lan:
            translation_response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"translate '{extracted_text}' to {target_language}",
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5
        )

            translated_text = translation_response.choices[0].text

            st.write("Translated Text:")
            st.write(translated_text)

            #translated text to speech conversion using gTTS

            lan = detect_language(translated_text)
            tts = gTTS(text=translated_text, lang=lan)
            tts.save("translated_speech.mp3")
            st.audio("translated_speech.mp3")
        



    



