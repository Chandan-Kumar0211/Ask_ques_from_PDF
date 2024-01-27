import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from backend import get_response

# Sidebar contents
with st.sidebar:
    st.title('💬 LLM based PDF-Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Fireworks-ai](https://fireworks.ai/)

    ''')
    add_vertical_space(3)

    st.write('NOTE: Currently you can only ask question from the pdf: 48lawsofpower')

st.title("Chat with the PDF:")

query = st.text_input("Question: ")

if query:
    response = get_response(query)

    st.header("Answer")
    st.write(response)

