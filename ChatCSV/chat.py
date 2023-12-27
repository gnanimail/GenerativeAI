import streamlit as st 
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()

openai_api_key = "***********************************"

def chat_csv(df, prompt):
      llm = OpenAI(api_token=openai_api_key)
      result = SmartDataframe(df, config={"llm": llm})
      response = result.chat(prompt)     
      return response

st.set_page_config(layout='wide')
st.title("Chat World Happiness Report using LLM")

data_ = st.file_uploader("Upload your csv file", type=['csv'])

if data_ is not None:
    col1, col2 = st.columns([1,1])

    with col1:            
            st.info("CSV Uploaded Successfully")
            data = pd.read_csv(data_)
            st.dataframe(data, use_container_width=True)

    with col2:
          st.info("Chat Below")

          input_text = st.text_area("Enter your query")
          
          if input_text is not None:
                st.button("Chat with CSV")
                st.info("Your Query: "+input_text)
                response = chat_csv(data, input_text)
                st.success(response)
