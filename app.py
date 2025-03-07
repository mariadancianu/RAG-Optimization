"""
Source: official streamlit documentation: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps
"""

import random
import time

import pandas as pd
import streamlit as st

from rag_optimization import (
    CustomRAG,
    convert_knowledge_base_to_langchain_docs,
    prompt_message,
)

st.title("RAG chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


def response_generator_rag(query):
    # TODO: implement a class for retrieval only, skipping the knowledge base creation

    df = pd.read_csv("dataset.csv")
    langchain_docs = convert_knowledge_base_to_langchain_docs(df)

    rag = CustomRAG(
        knowledge_base=langchain_docs, prompt_message=prompt_message, save_results=False
    )

    answer, _ = rag.get_llm_single_question_answer(query)

    return answer


# React to user input
# assign user input to prompt variable and check that it's not None, in the same line
if user_query := st.chat_input("Type your message ✍"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_query)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator())

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
