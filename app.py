"""
Source: official streamlit documentation: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps
"""

import random
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag_optimization import CustomRAG, convert_knowledge_base_to_langchain_docs

prompt_message = f"""
You are a highly accurate and reliable assistant. Answer the user's question using **only** the provided context.
If the answer is not in the context, return "I'm afraid I can't assist with that. Is there something else you'd like to know?" without making up information.
If the user does not ask a question, do not respond with "I'm afraid I can't assist with that." Instead, provide a more conversational or appropriate response based on the input.

Context:
%s

Instructions:
- Answer concisely and precisely, but make the response sound like a natural conversation.
- If the answer is explicitly stated in the context, rephrase or elaborate on it in a conversational way.
- If the user does not ask a question, do not use the response "I'm afraid I can't assist with that." Instead, engage in a relevant conversation based on the input.
- If the answer is not in the context, return "I'm afraid I can't assist with that. Is there something else you'd like to know?"
- You may elaborate on the context when necessary to provide a more detailed answer, but **do not** infer or add new information.
- Do **not** assume anything beyond the provided context.

Example:
    **Question:** What is the capital of Italy?
    **Answer:** The capital of Italy is Rome, a city known for its rich history and cultural landmarks.

    **Statement:** I had a great day today.
    **Answer:** I'm glad to hear that! What made your day great?

Question: %s
Answer (in a conversational tone, with "I'm afraid I can't assist with that. Is there something else you'd like to know?" if unknown):
"""


st.title("RAG chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I assist you today?"}
    ]

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

    # Display spinner while waiting for the response
    with st.spinner("Thinking..."):
        # Get assistant response
        response = response_generator_rag(user_query)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
