from typing import Set

from backend.core import run_llm

import streamlit as st
from PIL import Image

# Page Configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Documentation Helper",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for UI Polish ---
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* User message background */
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #2b313e;
    }
    
    /* Assistant message background */
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #1e2329;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #171b21;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #f0f2f6;
        font-family: 'Inter', sans-serif;
    }
    
    /* Input field styling */
    .stChatInputContainer {
        padding-bottom: 1rem;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0e1117; 
    }
    ::-webkit-scrollbar-thumb {
        background: #555; 
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #888; 
    }
</style>
""", unsafe_allow_html=True)

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "\n\n**Sources:**\n"

    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

def render_user_sidebar():
    """Render the user information sidebar"""
    with st.sidebar:
        st.markdown("## 👤 User Profile")
        
        # Initialize user info in session state if not present
        if "user_name" not in st.session_state:
            st.session_state["user_name"] = ""
        if "user_email" not in st.session_state:
            st.session_state["user_email"] = ""
        if "user_profile_picture" not in st.session_state:
            st.session_state["user_profile_picture"] = None
        
        # Profile picture section
        st.subheader("Profile Picture")
        uploaded_file = st.file_uploader(
            "Upload profile picture",
            type=['png', 'jpg', 'jpeg'],
            key="profile_upload",
            help="Upload a profile picture (PNG, JPG, or JPEG)"
        )
        
        if uploaded_file is not None:
            # Read and display the uploaded image
            image = Image.open(uploaded_file)
            st.session_state["user_profile_picture"] = image
            st.image(image, width=150, use_container_width=True)
        elif st.session_state["user_profile_picture"] is not None:
            # Display existing profile picture
            st.image(st.session_state["user_profile_picture"], width=150, use_container_width=True)
        else:
            # Display placeholder
            st.info("No profile picture uploaded")
        
        st.divider()
        
        # User information form
        st.subheader("User Information")
        name = st.text_input(
            "Name",
            value=st.session_state["user_name"],
            key="name_input",
            placeholder="Enter your name"
        )
        st.session_state["user_name"] = name
        
        email = st.text_input(
            "Email",
            value=st.session_state["user_email"],
            key="email_input",
            placeholder="Enter your email"
        )
        st.session_state["user_email"] = email
        
        st.divider()
        
        # Display current user info summary
        if st.session_state["user_name"] or st.session_state["user_email"]:
            st.subheader("Current Profile")
            if st.session_state["user_name"]:
                st.write(f"**Name:** {st.session_state['user_name']}")
            if st.session_state["user_email"]:
                st.write(f"**Email:** {st.session_state['user_email']}")

def main():
    # Render sidebar first
    render_user_sidebar()
    
    st.title("📚 Documentation Helper")
    st.markdown("*Ask me anything about your documentation!*")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    # Display chat history
    for message in st.session_state["chat_history"]:
        role = message[0]
        content = message[1]
        
        # Map 'human' to 'user' and 'ai' to 'assistant' for st.chat_message
        display_role = "user" if role == "human" else "assistant"
        
        with st.chat_message(display_role):
            st.markdown(content)

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add to chat history (so it persists)
        st.session_state["chat_history"].append(("human", prompt))
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner('Thinking...'):
                try:
                    
                    past_history = st.session_state["chat_history"][:-1]
                    generated_response = run_llm(
                        query=prompt, chat_history=past_history
                    )
                    
                    sources = set(
                        [doc.metadata["source"] for doc in generated_response["source_documents"]]
                    )
                    
                    result_text = generated_response['result']
                    sources_text = create_sources_string(sources)
                    formatted_response = f"{result_text} {sources_text}"
                    
                    st.markdown(formatted_response)
                    
                    # Add AI response to history
                    st.session_state["chat_history"].append(("ai", formatted_response))
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
