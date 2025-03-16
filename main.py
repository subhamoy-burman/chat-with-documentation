from typing import Set
from backend.core import run_llm
import streamlit as st

st.header("Langchain Documentation Helper Bot")

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return "No sources found"
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

prompt = st.text_input("prompt", placeholder="Enter your prompt here...")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answer_history" not in st.session_state:
    st.session_state["chat_answer_history"] = []

if prompt:
    with st.spinner("Thinking..."):
        generated_response = run_llm(prompt)
        sources = set([doc.metadata["source"] for doc in generated_response["source_document"]])

        formatted_response = (f"{generated_response["result"]} \n\n {create_sources_string(sources)}")
        print(formatted_response)

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answer_history"].append(formatted_response)


if(st.session_state["chat_answer_history"]):
    for generated_response, user_query in zip(st.session_state["chat_answer_history"],st.session_state["user_prompt_history"]):
       st.chat_message("user").write(user_query)
       st.chat_message("assistant").write(generated_response)
    
