from typing import Set

from backend.core import run_llm

import streamlit as st
from PIL import Image


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"

    for i, source in enumerate(sources_list):
        sources_string += f"{1+1}. {source}\n"
    return sources_string

def render_user_sidebar():
    """Render the user information sidebar"""
    with st.sidebar:
        st.header("👤 User Profile")
        
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
    
    st.header("Documentation Helper Bot")

    prompt = st.text_input("Prompt", placeholder="Enter your prompt here...")

    if (
            "user_prompt_history" not in st.session_state
            and "chat_answer_history" not in st.session_state
            and "chat_history" not in st.session_state
    ):
        st.session_state["user_prompt_history"] = []
        st.session_state["chat_answer_history"] = []
        st.session_state["chat_history"] = []

    if prompt:
        with st.spinner('Processing'):
            generated_response = run_llm(
                query=prompt, chat_history=st.session_state["chat_history"]
            )
            sources = set(
                [doc.metadata["source"] for doc in generated_response["source_documents"]]
            )

            formatted_response = (
                f"{generated_response['result']} \n\n {create_sources_string(sources)}"
            )

            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answer_history"].append(formatted_response)
            st.session_state["chat_history"].append(("human", prompt))
            st.session_state["chat_history"].append(("ai",generated_response["result"]))
    if st.session_state["chat_answer_history"]:
        for generated_response, user_query in zip(st.session_state["chat_answer_history"], st.session_state["user_prompt_history"]):
            st.chat_message("user").write(user_query)
            st.chat_message("assistant").write(generated_response)


if __name__ == "__main__":
    main()
