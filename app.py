"""
Source: official streamlit documentation: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps
Source login: https://medium.com/@kuldeepkumawat195/building-a-secure-user-authentication-system-with-streamlit-and-sqlite-9cf153288e18
"""

__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import hashlib
import random
import sqlite3
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# print(f"SQLite version: {sqlite3.sqlite_version}")

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


def init_db():
    """
    Use a lightweight disk-based database to store user credentials.
    """

    # Connection (and creation if not existent) to the on-disk database
    conn = sqlite3.connect("users.db")

    # In order to execute SQL statements and fetch results from SQL queries,
    # we will need to use a database cursor (a pointer)
    cursor = conn.cursor()

    # Creates the users table if it doesn't exist
    # with columns: username, password, role
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user'
        )
    """
    )

    # Commit any pending transaction to the database
    conn.commit()

    # Close the database connection
    conn.close()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def add_user_to_db(username, hashed_password):
    """
    Add a new user to the database.
    """

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hashed_password),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        st.error("Username already exists. Please choose another.")
    conn.close()


def signup():
    """
    Sign up for a new account if the "Sign Up" button is selected.
    """

    new_username = st.text_input("Create Username")
    new_password = st.text_input("Create Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if new_password != confirm_password:
            st.error("Passwords do not match.")
        else:
            hashed_password = hash_password(new_password)
            add_user_to_db(new_username, hashed_password)
            st.success("Account created successfully!")
            st.session_state["logged_in"] = True
            st.rerun()  # Force a rerun to refresh the UI


def verify_user(username, password):
    """
    Verify the user's credentials against the database.
    """

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    hashed_password = hash_password(password)
    cursor.execute(
        "SELECT role FROM users WHERE username = ? AND password = ?",
        (username, hashed_password),
    )
    result = cursor.fetchone()
    conn.close()
    return (result is not None, result[0] if result else None)


def login():
    """
    Login to the app if the "Login" button is selected. Verify the user's
    credentials against the database.

    """

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        is_valid, role = verify_user(username, password)
        if is_valid:
            st.success(f"Welcome back, {username}!")
            st.session_state["role"] = role
            st.session_state["username"] = username
            st.session_state["logged_in"] = True
            st.rerun()  # Force a rerun to refresh the UI
        else:
            st.error("Invalid username or password")


def chatbot():
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
            knowledge_base=langchain_docs,
            prompt_message=prompt_message,
            save_results=False,
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


# Main function to handle different states
def main():
    st.sidebar.title("Navigation")

    # Check if the user is logged in
    if st.session_state.get("logged_in", False):
        # If logged in, show the logout button and the chatbot
        logout = st.sidebar.button(label="Log Out")
        if logout:
            st.session_state.logged_in = False  # Set logged_in to False on logout

            # delete the chatbot messages when logging out
            if "messages" in st.session_state:
                del st.session_state.messages

            st.rerun()  # Rerun the app to refresh the state
        chatbot()  # Show the chatbot interface
    else:
        # If not logged in, show the login/signup options
        choice = st.sidebar.radio("Go to", ["Login", "Sign Up"])
        if choice == "Sign Up":
            signup()
        elif choice == "Login":
            login()


# Initialize database on first run
if __name__ == "__main__":
    init_db()
    main()
