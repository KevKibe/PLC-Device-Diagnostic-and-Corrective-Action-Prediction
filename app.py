import streamlit as st
import json
from keras.preprocessing.text import tokenizer_from_json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


def generate_text(model, tokenizer, input_text, max_length=40):
    generated_text = input_text
    stop_condition = False
    while not stop_condition:
        # tokenize the input texta
        input_sequence = tokenizer.texts_to_sequences([generated_text])[0]
        # pad the input sequence
        input_sequence = pad_sequences([input_sequence], maxlen=max_length-1, padding='pre')
        # make a prediction
        prediction = model.predict(input_sequence)[0]
        # get the index of the predicted word
        predicted_index = np.argmax(prediction)
        # get the predicted word
        predicted_word = tokenizer.index_word.get(predicted_index, '')
        # check if we've generated the maximum length or found the end token
        if len(generated_text.split()) == max_length or predicted_word == 'end':
            stop_condition = True
        else:
            # append the predicted word to the generated text
            generated_text += ' ' + predicted_word
    return generated_text[len(input_text):]
  
  
#loading tokenizer  
with open('/model/plc_lstm_tokenizer (2).json', 'r') as f:
    json_string = f.read()

tokenizer_json = json.loads(json_string)
tokenizer= tokenizer_from_json(tokenizer_json)
#loading model
model = tf.keras.models.load_model('/model/plc_lstm_14_model.h5')

st.title("PLC Device Diagnostic and Corrective Action Prediction')
         
text1=  st.text_input('Enter the PLC')
text2=  st.text_input('Enter the Model')
text3=  st.text_input('Enter the Fault')
input_text = text1 + ' ' + text2 + ' ' + text3
generate_button = st.button('Generate')

if generate_button:
   generated_text = generate_text(model, tokenizer, input_text)
   st.write('Diagnostic and Corrective Action:', generated_text)



