import streamlit as st
import requests
from requests.auth import HTTPBasicAuth

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "role" not in st.session_state:
    st.session_state["role"] = ""
if "login_error" not in st.session_state:
    st.session_state["login_error"] = None
    
# Protect route
if not st.session_state.get("authenticated"):
    st.warning("You must login first.")
    st.stop()

st.title("Chatbot Interface")
st.write(f"Welcome **{st.session_state['username']}**! Your role is **{st.session_state['role']}**.")

if st.button("Log-out"):
    
    st.session_state.clear()
    st.switch_page("stream_app.py")  

@st.cache_data
def query_backend(query, role):
    response = requests.post(
        url="http://localhost:8000/chat",
        json={"message": query, "role": role}
    )
    return response.json()
    
if st.session_state.get('load', False):
    # Textbox for query input
    query = st.text_input("Enter your query")
    url = "http://localhost:8000/chat"
    
    try:
        if query:
            data = query_backend(query,st.session_state['role'])
            if data:
                st.write(data["response"])
        
    except requests.exceptions.HTTPError:
        st.session_state['authenticated'] = False
        st.session_state['login_error'] = "Invalid credentials. Please try again."
        
else:
    st.warning("Please load data first.")
    st.stop()