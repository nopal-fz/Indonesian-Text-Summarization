import streamlit as st
from transformers import BertTokenizer, EncoderDecoderModel, EncoderDecoderConfig
import PyPDF2

model_ckpt = 'ardavey/bert2gpt-indosum'
tokenizer = BertTokenizer.from_pretrained(model_ckpt)
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

config = EncoderDecoderConfig.from_pretrained(model_ckpt)
config.early_stopping = True

model = EncoderDecoderModel.from_pretrained(model_ckpt, config=config)

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def summarize_text(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

st.title('Aplikasi Peringkasan Teks')

text_input = st.text_area('Masukkan teks atau unggah PDF')
if st.button('Ringkas Teks'):
    summary = summarize_text(text_input)
    st.write('Ringkasan:', summary)
