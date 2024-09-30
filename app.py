import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import os
import time

# Load model and tokenizer from file .pkl in the checkpoint folder
checkpoint_dir = './checkpoint'
model_path = os.path.join(checkpoint_dir, 'best_model.pkl')
tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer.pkl')

# Load tokenizer
with open(tokenizer_path, "rb") as f:
    tokenizer = torch.load(f, map_location=torch.device('cpu'), weights_only=False)

# Initialize model with the same architecture
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-xsum')

# Load state dict into the model with `weights_only=True`
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))

# Move model to CPU
device = torch.device("cpu")
model.to(device)

# Function to summarize text
def summarize_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = inputs.to(device)
    summary_ids = model.generate(inputs.input_ids, max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit UI
st.title('Dialogue Summarization App')

# User input for the dialogue
dialogue = st.text_area('Enter the dialogue to summarize')

# Button to generate summary
if st.button('Summarize'):
    if dialogue:
        with st.spinner('Generating summary...'):
            start_time = time.time()  # Start timer
            summary = summarize_text(dialogue)
            end_time = time.time()  # End timer
            execution_time = end_time - start_time  # Calculate execution time
        st.success('Summary:')
        st.write(summary)
        st.info(f"Execution Time: {execution_time:.2f} seconds")  # Display execution time
    else:
        st.error('Please enter a dialogue to summarize')
