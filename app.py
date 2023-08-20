import streamlit as st
from streamlit_chat import message
from model import load_llm
from data_loader import load_csv_data
from embedding import generate_and_save_embeddings
from langchain.chains import ConversationalRetrievalChain


def conversational_chat(chain, query, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def main():
    st.title("Chat with CSV using Llama2 🦙🦜")
    st.markdown("<h3 style='text-align: center; color: white;'>Built by Prajna</h3>", unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

    if uploaded_file:
        data = load_csv_data(uploaded_file)
        db = generate_and_save_embeddings(data)
        
        llm = load_llm()
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

        # Initializations
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! Ask me anything about " + uploaded_file.name + " 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey! 👋"]

        # Create containers for chat history and user input
        response_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                
                user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')
                
            if submit_button and user_input:
                output = conversational_chat(chain, user_input, st.session_state['history'])
                
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

if __name__ == "__main__":
    main()
