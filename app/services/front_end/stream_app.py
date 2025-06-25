import os
import streamlit as st
import requests
from requests.auth import HTTPBasicAuth



# Streamlit page setup
st.set_page_config(page_title="Role Based Chatbot")
st.title("Role Based Chatbot")



# Load data only once
if 'load' not in st.session_state and st.session_state.get('load', False) is False:
    with st.spinner("Loading data..."):
        try:
            url = "http://localhost:8000/load_data"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            st.session_state['load'] = True
            st.success("Data loaded successfully!")

        except requests.exceptions.HTTPError:
            st.error("Data loading Failed!")
            st.session_state['load'] = False
            


# --- User Validation ---
def validate_user():
    if st.session_state.get('load', False) is False:
        st.warning("Please load data first.")
    else:
        username = st.session_state.get('login_username')
        password = st.session_state.get('login_password')

        if not username or not password:
            st.session_state['login_error'] = "Please enter both username and password."
            return

        url = "http://localhost:8000/login"

        try:
            response = requests.get(url, auth=HTTPBasicAuth(username, password))
            response.raise_for_status()
            data = response.json()

            st.session_state['authenticated'] = True
            st.session_state['role'] = data.get('role')
            st.session_state['username'] = username
            st.session_state['password'] = password
            st.session_state['login_error'] = None

        except requests.exceptions.HTTPError:
            st.session_state['authenticated'] = False
            st.session_state['login_error'] = "Invalid credentials. Please try again."


# --- Page Switch if Authenticated ---
if st.session_state.get("authenticated"):
    st.switch_page("pages/chatbot_page.py")


# --- Login Form ---
with st.form("login_form"):
    st.write("Please Login")
    st.text_input("Enter your username", key="login_username")
    st.text_input("Enter your password", type="password", key="login_password")
    st.form_submit_button("Submit", on_click=validate_user)

# Show login errors if any
if st.session_state.get("login_error"):
    st.error(st.session_state["login_error"])
