import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings

def get_pdf_text(pdf_docs):
    if pdf_docs is None or len(pdf_docs) == 0:
        return ""
    
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    if not text:
        return []

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    if not text_chunks:
        return None
    
    # Initialize the HuggingFaceEmbeddings model with a better model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    
    # Use FAISS to create a vector store
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    
    return vectorstore

def get_conversation_chain(vectorstore):
    if vectorstore is None:
        return None
    
    # Use a more advanced LLM from HuggingFaceHub
    llm = HuggingFaceHub(repo_id="microsoft/DialoGPT-medium", task="conversational", model_kwargs={"temperature": 0.7, "max_length": 1024})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        return
    
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            detailed_response = message.content
            if len(detailed_response.split()) < 20:  # Check if the response is too short
                detailed_response += " Please provide more details or clarify your question."
            st.write(bot_template.replace("{{MSG}}", detailed_response), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask AVA", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Ask AVA, a chatbot to answer query related to your documents :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                if not text_chunks:
                    st.warning("No text found in the uploaded PDFs.")
                    return
                
                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                if vectorstore is None:
                    st.error("Failed to create vector store.")
                    return

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
